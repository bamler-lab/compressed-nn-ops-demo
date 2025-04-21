struct Globals {
  cursor: u32,
  input_dim: u32,
  output_dim: u32,
}

@group(0) @binding(0)
var<uniform> globals: Globals;

@group(0) @binding(1)
var<storage, read> matrix: array<u32>;

@group(0) @binding(2)
var<storage, read> input_vector: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output_vector: array<u32>;

var<workgroup> input_vector_workgroup: array<u32, 1024>; // TODO: can we make this run-time sized?

@compute @workgroup_size(64)
fn mat_vec_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    var cursor = globals.cursor;
    var word = matrix[cursor];
    let grid_spacing = unpack2x16float(word).x;
    cursor += 1;

    // Copy `input_vector` into workgroup memory.
    // Assume that `globals.input_dim` is a multiple of 4.
    let workgroup_size = 64u;
    var start = local_id.x * ((globals.input_dim / 4 + workgroup_size - 1) / workgroup_size);
    var end = min(start + workgroup_size, globals.input_dim / 4);
    for (var i = start; i != end; i += 1u) {
        input_vector_workgroup[i] = input_vector[i];
    }
    workgroupBarrier();

    if (global_id.x >= globals.output_dim) {
        return;
    }
    
    start = cursor + global_id.x * (globals.input_dim / 4);
    let count = globals.input_dim / 4;
    var accumulator = 0i;
    for (var i = 0u; i != count; i += 1u) {
        accumulator += dot4I8Packed(input_vector_workgroup[i], matrix[start + i]);
    }

    let result = u32(i32(round(f32(accumulator) * grid_spacing))) & 0xff;
    let next_result = subgroupShuffleDown(result, 1u);
    let pair = (next_result << 8) | result;
    let next_pair = subgroupShuffleDown(pair, 2u);
    if (global_id.x % 4 == 0) {
        output_vector[global_id.x / 4] = (next_pair << 16) | pair;
    }
}
