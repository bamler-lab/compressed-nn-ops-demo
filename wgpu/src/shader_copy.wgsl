@group(0) @binding(0)
var<uniform> dim: u32;

@group(0) @binding(1)
var<storage, read> source: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dest: array<u32>;

@compute @workgroup_size(64)
fn copy(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    if (global_id.x < dim) {
        dest[global_id.x] = source[global_id.x];
    }
}
