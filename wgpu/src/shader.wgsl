struct Globals {
  cursor: u32,
  input_dim: u32,
  output_dim: u32,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(0) @binding(1)
var<storage, read> compressed_data: array<u32>;

@group(0) @binding(2)
var<storage, read> input_vector: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output_vector: array<u32>;

var<workgroup> ppf: array<u32, 256>;

var<workgroup> input_vector_workgroup: array<u32, 1024>; // TODO: can we make this run-time sized?

// TODO:
// - use a 64 bit state and decode 4 values at a time
// - unpack the lowest 32 bit into 4 quantiles (using `unpack4xU8`), perform lookups in parallel if possible
// - calculate state update as much as possible in 32 bit, require only one 64 bit MAC
// - use `dot4I8Packed` to compute the dot product of length-4 slices
// - then use `subgroupAdd` to sum the results of the dot products across a subgroup
// - use pointers instead of array indices

// LATER:
// - interleave compressed data within each subgroup and use `subgroupInclusiveAdd` to calculate cursor increments

// Ideal workgroup size depends on the hardware, the workload, and other factors. However, it should
// _generally_ be a multiple of 64. Common sizes are 64x1x1, 256x1x1; or 8x8x1, 16x16x1 for 2D workloads.
@compute @workgroup_size(64)
fn mat_vec_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    // @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    // `global_invocation_id.x` ranges from 0 to 4095
    // `subgroup_id` is [16 times 0, 16 times 1, 16 times 2, 16times 3, then repeats: 16 times 0 ...]
    // --> probably identifies the subgroup id *within* the workgroup, where each workgroup contains 4 subgroups.
    // --> on work PC, it's just 0.
    // `subgroup_invocation_id` is [0, 1, 2, ..., 15, then repeats]
    // --> on work PC, it's [0, 1, 2, ..., 63, then repeats]
    // `subgroup_size` is 16
    // --> on work PC, it's 64

    // Create PPF lookup table in workgroup memory. Each instance within a workgroup
    // is responsible for filling entries ppf[(4 * local_id.x)..(4 * local_id.x + 4)].

    var quantile = 4 * local_id.x;
    var cursor = globals.cursor;
    var word = compressed_data[cursor];
    let grid_spacing = unpack2x16float(word).x;
    let grid_start = i32(((word >> 16) + 128) & 0xFF) - 128;
    let grid_size = (word >> 24) & 0xFF;
    cursor += 1;
    // The left-sided cdf ranges from `compressed_data[cursor][0]` (inclusively) to 
    // `compressed_data[cursor + grid_size / 4][grid_size % 4]` (exclusively), where
    // the second `[]` denotes zero-based indexing of the byte in a 4-byte word.

    // TODO: Maybe copy `cdf` into workgroup memory first (can be written to `ppf`
    // buffer and then overwritten after a workgroup barrier).

    // Find the last cdf entry that is <= quantile:
    // Assume that grid_size <= 128.
    let max_search_idx = (grid_size - 1u) / 4;
    var search_idx = 0u;
    var bit = 16u;

    for (var i = 0u; i!=5; i+=1) {
        let test_cdf = compressed_data[cursor + (search_idx | bit)] & 0xFF;
        let is_leq = (test_cdf <= quantile) && ((search_idx | bit) <= max_search_idx);
        search_idx |= bit * u32(is_leq);
        bit >>= 1;
    }

    word = compressed_data[cursor + search_idx];
    var test_cdf = (word >> 16) & 0xFF;
    var is_leq = (test_cdf <= quantile) && (((search_idx * 4) | 2) < grid_size);
    var sub_search_index = 2 * u32(is_leq);
    
    test_cdf = (word >> ((sub_search_index | 1) * 8)) & 0xFF;
    is_leq = (test_cdf <= quantile) && (((search_idx * 4) | sub_search_index | 1) < grid_size);
    sub_search_index |= u32(is_leq);

    var cdf_cursor = (search_idx * 4) | sub_search_index;
    var left_cdf = (compressed_data[cursor + cdf_cursor / 4] >> (8 * (cdf_cursor % 4))) & 0xFF;
    cdf_cursor += 1;
    var right_cdf = (compressed_data[cursor + cdf_cursor / 4] >> (8 * (cdf_cursor % 4))) & 0xFF;
    var ppf_entry = pack4xU8(vec4(
        left_cdf,
        right_cdf - left_cdf, // `pack4xU8` will truncate this to 8 bits
        cdf_cursor - 1,
        0
    ));
    ppf[quantile] = ppf_entry;

    let end_quantile = 4 * local_id.x + 3;
    while (quantile != end_quantile) {
        quantile += 1;
        if (right_cdf == quantile) {
            cdf_cursor += 1;
            left_cdf = right_cdf;
            right_cdf = (compressed_data[cursor + cdf_cursor / 4] >> (8 * (cdf_cursor % 4))) & 0xFF;
            ppf_entry = pack4xU8(vec4(
                left_cdf,
                right_cdf - left_cdf, // `pack4xU8` will truncate this to 8 bits
                cdf_cursor - 1,
                0
            ));
        }
        ppf[quantile] = ppf_entry;
    }


    // Copy `input_vector` into workgroup memory.
    // Assume that `globals.input_dim` is a multiple of 4.
    let workgroup_size = 64u;
    let start = local_id.x * ((globals.input_dim / 4 + workgroup_size - 1) / workgroup_size);
    let end = min(start + workgroup_size, globals.input_dim / 4);
    for (var i = start; i != end; i += 1u) {
        input_vector_workgroup[i] = input_vector[i];
    }
    workgroupBarrier();

    if (global_id.x >= globals.output_dim) {
        return;
    }

    // Initialize decoder state.

    let offsets_start = globals.cursor + 1 + (grid_size + 4) / 4;
    let offset = compressed_data[offsets_start + global_id.x / subgroup_size];
    let subgroup_start = offsets_start + globals.output_dim / subgroup_size + 1 + offset;
    cursor = subgroup_start + 2 * subgroup_invocation_id;

    var state = u64(compressed_data[cursor]) << 32;
    cursor += 1;
    state |= u64(compressed_data[cursor]);
    let debug_initial_state = state;

    cursor = subgroup_start + 2 * subgroup_size; // Skip all initial states of the subgroup.

    var accumulator = 0i;

    for (var col = 0u; col != globals.output_dim / 4; col += 1u) {
        let lower_state = u32(state);
        state >>= 32;

        let quantiles = unpack4xU8(lower_state);

        let ppf_entry0 = unpack4xU8(ppf[quantiles[0]]);
        let ppf_entry1 = unpack4xU8(ppf[quantiles[1]]);
        let ppf_entry2 = unpack4xU8(ppf[quantiles[2]]);
        let ppf_entry3 = unpack4xU8(ppf[quantiles[3]]);

        let matrix_entries = vec4(
            i32(ppf_entry0[2]) + grid_start,
            i32(ppf_entry1[2]) + grid_start,
            i32(ppf_entry2[2]) + grid_start,
            i32(ppf_entry3[2]) + grid_start,
        );

        // TODO: This emulates `dot4I8Packed`, which doesn't seem to be available.
        // Do we need to request some feature to use it?
        let input_vector_entries = unpack4xI8(input_vector_workgroup[col]);
        accumulator += dot(matrix_entries, input_vector_entries);

        // TODO: maybe do this as a single 32-bit subtraction.
        let remainder0 = quantiles[0] - ppf_entry0[0];
        let remainder1 = quantiles[1] - ppf_entry1[0];
        let remainder2 = quantiles[2] - ppf_entry2[0];
        let remainder3 = quantiles[3] - ppf_entry3[0];

        var full_remainder = remainder0 * ppf_entry1[1];
        full_remainder += remainder1;
        full_remainder *= ppf_entry2[1];
        full_remainder += remainder2;
        full_remainder *= ppf_entry3[1];
        full_remainder += remainder3;

        let full_probability = ppf_entry0[1] * ppf_entry1[1] * ppf_entry2[1] * ppf_entry3[1];

        state = u64(full_probability) * state + u64(full_remainder);

        let needs_refill = state >> 32 == 0;
        if (needs_refill) {
            state = (state << 32) | u64(compressed_data[cursor + subgroupExclusiveAdd(u32(needs_refill))]);
        }
        cursor += subgroupAdd(u32(needs_refill));
    }

    let result = u32(i32(round(f32(accumulator) * grid_spacing))) & 0xff;

    let next_result = subgroupShuffleDown(result, 1u);
    let pair = (next_result << 8) | result;
    let next_pair = subgroupShuffleDown(pair, 2u);
    if (global_id.x % 4 == 0) {
        output_vector[global_id.x / 4] = (next_pair << 16) | pair;
    }
}
