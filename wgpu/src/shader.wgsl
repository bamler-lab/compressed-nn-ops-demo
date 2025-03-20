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
var<storage, read> input_vector: array<i32>;

@group(0) @binding(3)
var<storage, read_write> output_vector: array<i32>;

var<workgroup> ppf: array<u32, 256>;

// TODO:
// - use a 64 bit state and decode 4 values at a time
// - unpack the lowest 32 bit into 4 quantiles (using `unpack4xU8`), perform lookups in parallel if possible
// - calculate state update as much as possible in 32 bit, require only one 64 bit MAC
// - use `dot4I8Packed` to compute the dot product of length-4 slices
// - then use `subgroupAdd` to sum the results of the dot products across a subgroup

// LATER:
// - interleave compressed data within each subgroup and use `subgroupInclusiveAdd` to calculate cursor increments

// Ideal workgroup size depends on the hardware, the workload, and other factors. However, it should
// _generally_ be a multiple of 64. Common sizes are 64x1x1, 256x1x1; or 8x8x1, 16x16x1 for 2D workloads.
@compute @workgroup_size(64)
fn mat_vec_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(subgroup_size) subgroup_size: u32
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
        cdf_cursor - 1, // TODO: add `grid_start` (also below)
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
    

    let num_rows = arrayLength(&output_vector);
    if (global_id.x >= num_rows) {
        return;
    }

    // var<workgroup> ppf: array<u32, 256>;

    if (global_id.x >= globals.output_dim - 64) {
        output_vector[globals.output_dim - 256 + 4 * local_id.x] = i32((ppf[4 * local_id.x] >> 16) & 0xFF);
        output_vector[globals.output_dim - 256 + 4 * local_id.x + 1] = i32((ppf[4 * local_id.x + 1] >> 16) & 0xFF);
        output_vector[globals.output_dim - 256 + 4 * local_id.x + 2] = i32((ppf[4 * local_id.x + 2] >> 16) & 0xFF);
        output_vector[globals.output_dim - 256 + 4 * local_id.x + 3] = i32((ppf[4 * local_id.x + 3] >> 16) & 0xFF);
        // output_vector[global_id.x] = i32((compressed_data[cursor + search_idx] >> (8 * sub_search_index)) & 0xFF);
    } else if (global_id.x < globals.output_dim - 256) {
        output_vector[global_id.x] = i32((compressed_data[cursor + 8] / 16) &0xff);
    }

    // output_vector[global_id.x] = i32(
    //     (compressed_data[cursor + global_id.x / 4] >> ((global_id.x % 4) * 8)) & 0xff
    // );
}
