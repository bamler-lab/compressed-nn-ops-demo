use std::{io::Read, num::NonZeroU64, path::PathBuf};

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

    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn main() -> Result<()> {
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

    let limits = adapter.limits();

    dbg!(limits.max_compute_workgroup_storage_size);
    dbg!(limits.max_compute_invocations_per_workgroup);
    dbg!(limits.max_compute_workgroup_size_x);
    dbg!(limits.max_compute_workgroup_size_y);
    dbg!(limits.max_compute_workgroup_size_z);
    dbg!(limits.max_compute_workgroups_per_dimension);
    dbg!(limits.min_subgroup_size);
    dbg!(limits.max_subgroup_size);
    dbg!(limits.max_push_constant_size);

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    dbg!(&downlevel_capabilities);
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,

            // TODO: try `PIPELINE_STATISTICS_QUERY` and `TIMESTAMP_QUERY_INSIDE_*`.
            // TODO: try `MAPPABLE_PRIMARY_BUFFERS` (with caution)
            // TODO (important): try `SUBGROUP` and `SUBGROUP_BARRIER`
            required_features: wgpu::Features::SUBGROUP
                | wgpu::Features::SUBGROUP_BARRIER
                | wgpu::Features::SHADER_INT64,
            required_limits: wgpu::Limits::downlevel_defaults(), // TODO: sets `{min, max}_subgroup_size` to 0
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    dbg!(&device);

    // TODO: try `create_shader_module_trusted` and turn off unnecessary runtime checks.
    let mut shader_file = std::fs::File::open("src/shader.wgsl")?;
    let mut shader_code = String::new();
    shader_file.read_to_string(&mut shader_code)?;
    // let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let globals = [
        0,                                // cursor
        file_header.dimensions[0] as u32, // input dimension
        file_header.dimensions[1] as u32, // output dimension
    ];
    let globals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&globals),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut compressed_data = vec![0; file_header.max_compressed_matrix_size as usize];
    let compressed_len =
        (file_header.offsets_and_file_size[1] - file_header.offsets_and_file_size[0]) as usize;
    reader.read_exact(&mut compressed_data[0..compressed_len])?;
    let compressed_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &compressed_data[0..compressed_len],
        usage: wgpu::BufferUsages::STORAGE,
    });

    let input_vector_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&input_vector.0),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // let ppf_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    //     label: None,
    //     size: header.input_vector.len() as u64,
    //     usage: wgpu::BufferUsages::STORAGE,
    //     mapped_at_creation: false,
    // });

    let output_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * file_header.dimensions[1] as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Finally we create a buffer which can be read by the CPU. This buffer is how we will read
    // the data. We need to use a separate buffer because we need to have a usage of `MAP_READ`,
    // and that usage can only be used with `COPY_DST`.
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * file_header.dimensions[1] as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // A bind group layout describes the types of resources that a bind group can contain. Think
    // of this like a C-style header declaration, ensuring both the pipeline and bind group agree
    // on the types of resources.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // `globals_buffer`
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    min_binding_size: Some(
                        NonZeroU64::new((globals.len() * size_of::<u32>()) as u64).unwrap(),
                    ),
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // `input_data_buffer`
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(size_of::<u32>() as u64).unwrap()), // TODO: what is this for?
                    has_dynamic_offset: false, // TODO: can we use this?
                },
                count: None,
            },
            // `input_vector_buffer`
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
            // `output_vector_buffer`
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
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: compressed_data_buffer.as_entire_binding(), // TODO: try `.slice()` instead.
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: input_vector_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
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

    // The command encoder allows us to record commands that we will later submit to the GPU.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // A compute pass is a single series of compute operations. While we are recording a compute
    // pass, we cannot record to the encoder.
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    // Set the pipeline that we want to use
    compute_pass.set_pipeline(&pipeline);
    // Set the bind group that we want to use
    compute_pass.set_bind_group(0, &bind_group, &[]);

    const WORKGROUP_SIZE: u32 = 64;
    let workgroup_count = file_header.dimensions[1].div_ceil(WORKGROUP_SIZE);
    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

    // Now we drop the compute pass, giving us access to the encoder again.
    drop(compute_pass);

    // We add a copy operation to the encoder. This will copy the data from the output buffer on the
    // GPU to the download buffer on the CPU.
    encoder.copy_buffer_to_buffer(
        &output_vector_buffer,
        0,
        &download_buffer,
        0,
        output_vector_buffer.size(),
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
    let result: &[i32] = bytemuck::cast_slice(&data);

    // Print out the result.
    println!("Result: {:?}", result);

    Ok(())
}
