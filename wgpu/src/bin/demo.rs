use std::{collections::HashMap, io::Read, num::NonZeroU64, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use compressed_mat_vec_mul::{FileHeader, UncompressedVector};
use log::info;
use wgpu::util::DeviceExt;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to a file containing compressed matrices and an initial vector.
    /// Use `mk-random` to create a file with mock data in the expected format.
    input: PathBuf,

    /// Write the input vector, all intermediate vectors, and the final result to the provided file
    /// in `safetensors` format. If the file already exists it will be overwritten. Otherwise, a new
    /// file will be created.
    #[arg(long)]
    debug: Option<PathBuf>,

    /// Matrices are stored in uncompressed form. This is meant for baseline performance testing.
    #[arg(long)]
    uncompressed: bool,

    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Globals {
    cursor: u32,
    input_dim: u32,
    output_dim: u32,
}

fn main() -> Result<()> {
    const WORKGROUP_SIZE: u32 = 64;

    let cli = Cli::parse();

    // wgpu uses `log` rather than `tracing`, so we'll use that too.
    env_logger::Builder::new()
        .filter_level(cli.verbose.log_level_filter())
        .init();

    let file = std::fs::File::open(&cli.input)?;
    let mut reader = std::io::BufReader::new(file);

    let file_header = FileHeader::from_read(&mut reader)?;
    let input_vector = UncompressedVector::from_read(&mut reader, file_header.dimensions[0])?;

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");

    // Print out some basic information about the adapter.
    info!("Running on Adapter: {:#?}", adapter.get_info());
    dbg!(adapter.limits());

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    dbg!(&downlevel_capabilities);
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    let compressed_len = file_header.offsets_and_file_size[file_header.num_matrices() as usize]
        - file_header.offsets_and_file_size[0];

    let mut required_limits = wgpu::Limits::downlevel_defaults();
    required_limits.max_storage_buffer_binding_size = compressed_len;
    required_limits.max_buffer_size = compressed_len as u64;
    required_limits.min_uniform_buffer_offset_alignment =
        adapter.limits().min_uniform_buffer_offset_alignment;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            // TODO: try `PIPELINE_STATISTICS_QUERY` and `TIMESTAMP_QUERY_INSIDE_*`.
            // TODO: try `MAPPABLE_PRIMARY_BUFFERS` (with caution)
            required_features: wgpu::Features::SUBGROUP
                | wgpu::Features::SUBGROUP_BARRIER
                | wgpu::Features::SHADER_INT64,
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    dbg!(&device);

    // TODO: try `create_shader_module_trusted` and turn off unnecessary runtime checks.
    let shader_path = if cli.uncompressed {
        "src/shader_uncompressed.wgsl"
    } else {
        "src/shader.wgsl"
    };
    let mut shader_file = std::fs::File::open(shader_path)?;
    let mut shader_code = String::new();
    shader_file.read_to_string(&mut shader_code)?;
    // let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let mut shader_file_copy = std::fs::File::open("src/shader_copy.wgsl")?;
    let mut shader_code_copy = String::new();
    shader_file_copy.read_to_string(&mut shader_code_copy)?;
    // let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
    let module_copy = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_code_copy.into()),
    });

    let globals_alignment = (size_of::<Globals>() as u32)
        .next_multiple_of(adapter.limits().min_uniform_buffer_offset_alignment);
    let mut globals = vec![0u8; (globals_alignment * file_header.num_matrices()) as usize];
    for (i, globals) in globals
        .chunks_exact_mut(globals_alignment as usize)
        .enumerate()
    {
        let globals: &mut [Globals] =
            bytemuck::cast_slice_mut(&mut globals[0..size_of::<Globals>()]);
        globals[0] = Globals {
            cursor: (file_header.offsets_and_file_size[i] - file_header.offsets_and_file_size[0])
                / 4,
            input_dim: file_header.dimensions[i],
            output_dim: file_header.dimensions[i + 1],
        };
    }

    let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("globals_buffer"),
        contents: &globals,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // TODO: is COPY_DST necessary?
    });

    let copy_dim_alignment = (size_of::<u32>() as u32)
        .next_multiple_of(adapter.limits().min_uniform_buffer_offset_alignment);
    let mut copy_dims = vec![0u8; (copy_dim_alignment * 2) as usize];
    bytemuck::cast_slice_mut(&mut copy_dims[0..4])[0] = file_header.dimensions[0];
    bytemuck::cast_slice_mut(
        &mut copy_dims[copy_dim_alignment as usize..copy_dim_alignment as usize + 4],
    )[0] = file_header.dimensions[file_header.num_matrices() as usize];

    let copy_dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("copy_dims_buffer"),
        contents: &copy_dims,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // TODO: is COPY_DST necessary?
    });

    let mut compressed_data = vec![0; compressed_len as usize];
    reader.read_exact(&mut compressed_data[..])?;
    let compressed_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compressed_data_buffer"),
        contents: &compressed_data,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_vector_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_vector_buffer"),
        contents: bytemuck::cast_slice(&input_vector.0),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let max_vec_dimension = *file_header
        .dimensions
        .iter()
        .max()
        .expect("input vector must be present");

    let vector_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vector_a_buffer"),
        size: max_vec_dimension as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let vector_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vector_b_buffer"),
        size: max_vec_dimension as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_vector_buffer"),
        size: file_header.dimensions[file_header.num_matrices() as usize] as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Finally we create a buffer which can be read by the CPU. This buffer is how we will read
    // the data. We need to use a separate buffer because we need to have a usage of `MAP_READ`,
    // and that usage can only be used with `COPY_DST`.
    // TODO: this comment is outdated
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download_buffer"),
        size: file_header.dimensions[1] as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout_copy =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy bind group layout"),
            entries: &[
                // `copy_dims_buffer`
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                        has_dynamic_offset: true,
                    },
                    count: None,
                },
                // source
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // destination
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

    // A bind group layout describes the types of resources that a bind group can contain. Think
    // of this like a C-style header declaration, ensuring both the pipeline and bind group agree
    // on the types of resources.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mul bind group layout"),
        entries: &[
            // `globals_buffer`
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    min_binding_size: Some(NonZeroU64::new(globals_alignment as u64).unwrap()),
                    has_dynamic_offset: true,
                },
                count: None,
            },
            // `compressed_data_buffer`
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // TODO: we could make this a uniform buffer, but I'm not sure if that would make it any faster
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(size_of::<u32>() as u64).unwrap()), // TODO: what is this for?
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // input vector
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()), // TODO: what is this for?
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // output vector
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()), // TODO: what is this for?
                    has_dynamic_offset: false,
                },
                count: None,
            },
        ],
    });

    // The bind group contains the actual resources to bind to the pipeline.
    //
    // Even when the buffers are individually dropped, wgpu will keep the bind group and buffers
    // alive until the bind group itself is dropped.
    let bind_group_even = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("even"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(globals_alignment as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: compressed_data_buffer.as_entire_binding(), // TODO: try `.slice()` instead.
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vector_a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: vector_b_buffer.as_entire_binding(),
            },
        ],
    });

    let bind_group_odd = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("odd"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &globals_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(globals_alignment as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: compressed_data_buffer.as_entire_binding(), // TODO: try `.slice()` instead.
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vector_b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: vector_a_buffer.as_entire_binding(),
            },
        ],
    });

    let bind_group_copy_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("copy_in"),
        layout: &bind_group_layout_copy,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &copy_dims_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(copy_dim_alignment as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_vector_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vector_a_buffer.as_entire_binding(),
            },
        ],
    });

    let bind_group_copy_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout_copy,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &copy_dims_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(copy_dim_alignment as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: [&vector_a_buffer, &vector_b_buffer]
                    [(file_header.num_matrices() % 2) as usize | cli.debug.is_some() as usize]
                    .as_entire_binding(),
                // resource: vector_b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_vector_buffer.as_entire_binding(),
            },
        ],
    });

    // The pipeline layout describes the bind groups that a pipeline expects
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[], // TODO
    });

    let pipeline_layout_copy = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout_copy],
        push_constant_ranges: &[], // TODO
    });

    // The pipeline is the ready-to-go program state for the GPU. It contains the shader modules,
    // the interfaces (bind group layouts) and the shader entry point.
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("mat_vec_mul"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), // TODO: set `zero_initialize_workgroup_memory = false`
        cache: None,                                                      // TODO
    });

    let pipeline_copy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout_copy),
        module: &module_copy,
        entry_point: Some("copy"),
        compilation_options: wgpu::PipelineCompilationOptions::default(), // TODO: set `zero_initialize_workgroup_memory = false`
        cache: None,                                                      // TODO
    });

    if let Some(debug_path) = cli.debug {
        let mut intermediate_vector = input_vector.clone();

        let mut debug_data = (0..file_header.num_matrices())
            .map(|k| {
                let input_vector_buffer_debug =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("input_vector_buffer_debug"),
                        contents: bytemuck::cast_slice(&intermediate_vector.0),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                let bind_group_copy_in_debug =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("copy_in_debug"),
                        layout: &bind_group_layout_copy,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &copy_dims_buffer,
                                    offset: 0,
                                    size: Some(NonZeroU64::new(copy_dim_alignment as u64).unwrap()),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input_vector_buffer_debug.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: vector_a_buffer.as_entire_binding(),
                            },
                        ],
                    });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // Copy the input vector into `vector_a` (not sure if this detour is necessary).
                let mut compute_pass_copy_in =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                compute_pass_copy_in.set_pipeline(&pipeline_copy);
                compute_pass_copy_in.set_bind_group(0, &bind_group_copy_in_debug, &[0]);
                let workgroup_count = file_header.dimensions[0].div_ceil(WORKGROUP_SIZE);
                compute_pass_copy_in.dispatch_workgroups(workgroup_count, 1, 1);
                drop(compute_pass_copy_in);

                // Perform a matrix-vector multiplication.
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group_even, &[k * globals_alignment]);
                let workgroup_count =
                    file_header.dimensions[k as usize + 1].div_ceil(WORKGROUP_SIZE);
                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                drop(compute_pass);

                // Copy the result to a buffer that is marked for `COPY_SRC` usage.
                let mut compute_pass_copy_out =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                compute_pass_copy_out.set_pipeline(&pipeline_copy);
                compute_pass_copy_out.set_bind_group(
                    0,
                    &bind_group_copy_out,
                    &[copy_dim_alignment],
                );
                let workgroup_count = file_header.dimensions[0].div_ceil(WORKGROUP_SIZE);
                compute_pass_copy_out.dispatch_workgroups(workgroup_count, 1, 1);
                drop(compute_pass_copy_out);

                // Download the result from the GPU to the CPU.
                encoder.copy_buffer_to_buffer(
                    &output_vector_buffer,
                    0,
                    &download_buffer,
                    0,
                    file_header.dimensions[file_header.num_matrices() as usize] as u64,
                );

                let command_buffer = encoder.finish();
                queue.submit([command_buffer]);
                let buffer_slice = download_buffer.slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
                device.poll(wgpu::Maintain::Wait);

                let data = buffer_slice.get_mapped_range();
                let result: &[i8] = &bytemuck::cast_slice(&data)
                    [..file_header.dimensions[(k + 1) as usize] as usize];
                intermediate_vector.0.resize(result.len(), 0);
                intermediate_vector.0.copy_from_slice(result);
                let result = UncompressedVector(result.to_vec()).into_owned_safetensor();
                drop(data);
                download_buffer.unmap();

                (format!("vector_{}", k + 1), result)
            })
            .collect::<HashMap<_, _>>();

        debug_data.insert("vector_0".to_string(), input_vector.into_owned_safetensor());

        safetensors::serialize_to_file(debug_data, &None, &debug_path)?;
    } else {
        for _ in 0..10 {
            let start_time = std::time::Instant::now();
            // The command encoder allows us to record commands that we will later submit to the GPU.
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            // Copy the input vector into `vector_a` (not sure if this detour is necessary).
            let mut compute_pass_copy_in =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute_pass_copy_in.set_pipeline(&pipeline_copy);
            compute_pass_copy_in.set_bind_group(0, &bind_group_copy_in, &[0]);
            let workgroup_count = file_header.dimensions[0].div_ceil(WORKGROUP_SIZE);
            compute_pass_copy_in.dispatch_workgroups(workgroup_count, 1, 1);
            drop(compute_pass_copy_in);

            let mut k = 0;
            loop {
                // Perform the first matrix-vector multiplication.
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group_even, &[k * globals_alignment]);
                let workgroup_count =
                    file_header.dimensions[k as usize + 1].div_ceil(WORKGROUP_SIZE);
                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                drop(compute_pass);

                k += 1;
                if k == file_header.num_matrices() {
                    break;
                }

                // Perform the second matrix-vector multiplication.
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group_odd, &[k * globals_alignment]);
                let workgroup_count =
                    file_header.dimensions[k as usize + 1].div_ceil(WORKGROUP_SIZE);
                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                drop(compute_pass);

                k += 1;
                if k == file_header.num_matrices() {
                    break;
                }
            }

            // Copy the result to a buffer that is marked for `COPY_SRC` usage.
            let mut compute_pass_copy_out =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            compute_pass_copy_out.set_pipeline(&pipeline_copy);
            compute_pass_copy_out.set_bind_group(0, &bind_group_copy_out, &[copy_dim_alignment]);
            let workgroup_count = file_header.dimensions[0].div_ceil(WORKGROUP_SIZE);
            compute_pass_copy_out.dispatch_workgroups(workgroup_count, 1, 1);
            drop(compute_pass_copy_out);

            // Download the result from the GPU to the CPU.
            encoder.copy_buffer_to_buffer(
                &output_vector_buffer,
                0,
                &download_buffer,
                0,
                file_header.dimensions[file_header.num_matrices() as usize] as u64,
            );

            // We finish the encoder, giving us a fully recorded command buffer.
            let command_buffer = encoder.finish();

            // At this point nothing has actually been executed on the gpu. We have recorded a series of
            // commands that we want to execute, but they haven't been sent to the gpu yet.
            //
            // Submitting to the queue sends the command buffer to the gpu. The gpu will then execute the
            // commands in the command buffer in order.
            queue.submit([command_buffer]);

            // We now map the download buffer so we can read it. Mapping tells wgpu that we want to read/write
            // to the buffer directly by the CPU and it should not permit any more GPU operations on the buffer.
            //
            // Mapping requires that the GPU be finished using the buffer before it resolves, so mapping has a callback
            // to tell you when the mapping is complete.
            let buffer_slice = download_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {
                // In this case we know exactly when the mapping will be finished,
                // so we don't need to do anything in the callback.
                // TODO: what does this mean?
            });

            // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU, so we would need
            // to rely on the callback to know when the buffer is mapped.
            device.poll(wgpu::Maintain::Wait);

            // We can now read the data from the buffer.
            let data = buffer_slice.get_mapped_range();
            let result: &[i8] = bytemuck::cast_slice(&data);
            dbg!(result[0]);
            // println!("Result: {:?}", result);
            drop(data);
            download_buffer.unmap();
            let end_time = std::time::Instant::now();

            let duration = end_time - start_time;
            let num_matrix_elements = file_header
                .dimensions
                .iter()
                .zip(file_header.dimensions.iter().skip(1))
                .map(|(a, b)| (a * b) as u64)
                .sum::<u64>();
            let throughput = num_matrix_elements as f64 / duration.as_secs_f64();
            println!(
                "Duration for {} matrix multiplications: {:?}",
                file_header.num_matrices(),
                duration
            );
            println!(
                "Throughput: {throughput:.4e} elements/second (for {num_matrix_elements:.4e} elements)"
            );
            println!();
        }
    }

    Ok(())
}
