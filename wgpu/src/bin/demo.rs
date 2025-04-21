use std::{
    borrow::Borrow,
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    num::NonZeroU64,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result};
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

    let mut file = BufReader::new(
        File::open(&cli.input)
            .with_context(|| format!("Loading input data from {}", cli.input.display()))?,
    );
    let file_header = FileHeader::from_read(&mut file)?;
    let input_vector = UncompressedVector::from_read(&mut file, file_header.dimensions[0])?;
    let compressed_len = file_header.offsets_and_file_size[file_header.num_matrices() as usize]
        - file_header.offsets_and_file_size[0];
    let mut compressed_data = vec![0; compressed_len as usize];
    file.read_exact(&mut compressed_data[..])?;
    drop(file);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");

    // Print out some basic information about the adapter.
    info!("Running on Adapter: {:#?}", adapter.get_info());
    info!("Adapter limits: {:?}", adapter.limits());

    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    info!("Downlevel capabilities: {:?}", &downlevel_capabilities);
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    let mut required_limits = wgpu::Limits::downlevel_defaults();
    required_limits.max_storage_buffer_binding_size = compressed_len;
    required_limits.max_buffer_size = compressed_len as u64;
    required_limits.min_uniform_buffer_offset_alignment =
        adapter.limits().min_uniform_buffer_offset_alignment;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        // TODO: try `PIPELINE_STATISTICS_QUERY` and `TIMESTAMP_QUERY_INSIDE_*`.
        // TODO: try `MAPPABLE_PRIMARY_BUFFERS` (with caution)
        required_features: wgpu::Features::SUBGROUP
            | wgpu::Features::SUBGROUP_BARRIER
            | wgpu::Features::SHADER_INT64,
        required_limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    info!("Device: {:?}", &device);

    let shader_path = if cli.uncompressed {
        "src/shader_uncompressed.wgsl"
    } else {
        "src/shader.wgsl"
    };

    let ops = MatVecMulOps::new(
        &device,
        &adapter,
        &file_header,
        shader_path,
        "src/shader_copy.wgsl",
        WORKGROUP_SIZE,
    )?;

    if let Some(debug_path) = cli.debug {
        let mut intermediate_vector = input_vector.clone();
        let runner = ops.bind_matrices(&compressed_data)?;

        let mut debug_data = (0..file_header.num_matrices())
            .map(|k| {
                let result = runner.run(&queue, &intermediate_vector.0, k..k + 1)?;
                let result: &[i8] = bytemuck::cast_slice(&result);
                intermediate_vector.0.resize(result.len(), 0);
                intermediate_vector.0.copy_from_slice(result);
                let result = UncompressedVector(result.to_vec()).into_owned_safetensor();
                Ok((format!("vector_{}", k + 1), result))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        debug_data.insert("vector_0".to_string(), input_vector.into_owned_safetensor());

        safetensors::serialize_to_file(debug_data, &None, &debug_path)?;
    } else {
        let num_matrix_elements = file_header
            .dimensions
            .iter()
            .zip(file_header.dimensions.iter().skip(1))
            .map(|(a, b)| (a * b) as u64)
            .sum::<u64>();

        println!("Total number of matrix elements: {num_matrix_elements:.2e}");
        println!(
            "Reporting mean ± standard error over 10 runs (after discarding 2 runs of warmup) ..."
        );

        let timings = (0..12)
            .map(|_| {
                let start_time = std::time::Instant::now();
                let runner = ops.bind_matrices(&compressed_data)?;
                let result = runner.run(&queue, &input_vector.0, 0..file_header.num_matrices())?;
                std::hint::black_box(result[0]);
                let end_time = std::time::Instant::now();
                Ok((end_time - start_time).as_secs_f64())
            })
            .skip(2)
            .collect::<Result<Vec<_>>>()?;

        let (duration_mean, duration_std_err) = stats(&timings);
        let (throughput_mean, throughput_std_err) =
            stats(timings.iter().map(|d| num_matrix_elements as f64 / d));
        println!(
            "Including upload:     duration = {:>5.1} ± {:>4.1} ms;  throughput = {:>5.2} ± {:>4.2} G elements/second",
            duration_mean * 1e3,
            duration_std_err * 1e3,
            throughput_mean * 1e-9,
            throughput_std_err * 1e-9,
        );

        let runner = ops.bind_matrices(&compressed_data)?;
        let timings = (0..12)
            .map(|_| {
                let start_time = std::time::Instant::now();
                let result = runner.run(&queue, &input_vector.0, 0..file_header.num_matrices())?;
                std::hint::black_box(result[0]);
                let end_time = std::time::Instant::now();
                Ok((end_time - start_time).as_secs_f64())
            })
            .skip(2)
            .collect::<Result<Vec<_>>>()?;

        let (duration_mean, duration_std_err) = stats(&timings);
        let (throughput_mean, throughput_std_err) =
            stats(timings.iter().map(|d| num_matrix_elements as f64 / d));
        println!(
            "Not including upload: duration = {:>5.1} ± {:>4.1} ms;  throughput = {:>5.2} ± {:>4.2} G elements/second",
            duration_mean * 1e3,
            duration_std_err * 1e3,
            throughput_mean * 1e-9,
            throughput_std_err * 1e-9,
        );
    }

    Ok(())
}

// Returns the mean and standard error.
fn stats<I>(iter: I) -> (f64, f64)
where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: Borrow<f64>,
{
    let iter = iter.into_iter();
    let len = iter.len();
    assert!(len >= 2);
    let (sum, second_moment) = iter.fold((0.0, 0.0), |(sum, second_moment), x| {
        let x = *x.borrow();
        (sum + x, second_moment + x * x)
    });

    let mean = sum / len as f64;
    let std_err = ((second_moment - sum * mean) / (len * (len - 1)) as f64).sqrt();

    (mean, std_err)
}

struct MatVecMulOps<'device, 'file_header> {
    device: &'device wgpu::Device,
    file_header: &'file_header FileHeader,
    workgroup_size: u32,
    mv_mul_bind_group_layout: wgpu::BindGroupLayout,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    globals_alignment: u32,
    copy_dim_alignment: u32,
    globals: Vec<u8>,
    copy_dims: Vec<u8>,
    vector_a_buffer: wgpu::Buffer,
    vector_b_buffer: wgpu::Buffer,
    output_vector_buffer: wgpu::Buffer,
    download_buffer: wgpu::Buffer,
    copy_pipeline: wgpu::ComputePipeline,
    mv_mul_pipeline: wgpu::ComputePipeline,
}

impl<'device, 'file_header> MatVecMulOps<'device, 'file_header> {
    fn new(
        device: &'device wgpu::Device,
        adapter: &wgpu::Adapter,
        file_header: &'file_header FileHeader,
        mv_mul_shader_path: &str,
        copy_shader_path: &str,
        workgroup_size: u32,
    ) -> Result<Self> {
        let max_dimension = *file_header
            .dimensions
            .iter()
            .max()
            .expect("input vector must be present");

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
                cursor: (file_header.offsets_and_file_size[i]
                    - file_header.offsets_and_file_size[0])
                    / 4,
                input_dim: file_header.dimensions[i],
                output_dim: file_header.dimensions[i + 1],
            };
        }

        let copy_dim_alignment = (size_of::<u32>() as u32)
            .next_multiple_of(adapter.limits().min_uniform_buffer_offset_alignment);
        let mut copy_dims = vec![0u8; (copy_dim_alignment * 2) as usize];
        bytemuck::cast_slice_mut(&mut copy_dims[0..4])[0] = file_header.dimensions[0];
        bytemuck::cast_slice_mut(
            &mut copy_dims[copy_dim_alignment as usize..copy_dim_alignment as usize + 4],
        )[0] = file_header.dimensions[file_header.num_matrices() as usize];

        let mut mv_mul_shader_file = File::open(mv_mul_shader_path).with_context(|| {
            format!("Loading mat-vec-mul shader code from {mv_mul_shader_path}")
        })?;
        let mut mv_mul_shader_code = String::new();
        mv_mul_shader_file.read_to_string(&mut mv_mul_shader_code)?;
        // let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        // TODO: try `create_shader_module_trusted` and turn off unnecessary runtime checks.
        let mv_mul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(mv_mul_shader_code.into()),
        });

        let mut copy_shader_file = File::open(copy_shader_path)
            .with_context(|| format!("Loading copy shader code from {copy_shader_path}"))?;
        let mut copy_shader_code = String::new();
        copy_shader_file.read_to_string(&mut copy_shader_code)?;
        // let module = device.create_shader_module(wgpu::include_wgsl!("../shader.wgsl"));
        let copy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(copy_shader_code.into()),
        });

        let vector_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_a_buffer"),
            size: max_dimension as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let vector_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vector_b_buffer"),
            size: max_dimension as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_vector_buffer"),
            size: max_dimension as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create a buffer that can be read by the CPU. It therefore needs to have a usage of
        // `MAP_READ`, and that usage can only be used with `COPY_DST` (not `STORAGE`), which is why
        // this needs to be a separated buffer.
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_buffer"),
            size: max_dimension as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let copy_bind_group_layout =
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

        let mv_mul_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mul bind group layout"),
                entries: &[
                    // `globals_buffer`
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            min_binding_size: Some(
                                NonZeroU64::new(globals_alignment as u64).unwrap(),
                            ),
                            has_dynamic_offset: true,
                        },
                        count: None,
                    },
                    // `matrices_buffer`
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true }, // TODO: we could make this a uniform buffer, but I'm not sure if that would make it any faster
                            // This is the size of a single element in the buffer.
                            min_binding_size: Some(
                                NonZeroU64::new(size_of::<u32>() as u64).unwrap(),
                            ), // TODO: what is this for?
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

        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[], // TODO
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&copy_pipeline_layout),
            module: &copy_module,
            entry_point: Some("copy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), // TODO: set `zero_initialize_workgroup_memory = false`
            cache: None,                                                      // TODO
        });

        let mv_mul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&mv_mul_bind_group_layout],
                push_constant_ranges: &[], // TODO
            });

        let mv_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&mv_mul_pipeline_layout),
            module: &mv_mul_module,
            entry_point: Some("mat_vec_mul"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), // TODO: set `zero_initialize_workgroup_memory = false`
            cache: None,                                                      // TODO
        });

        Ok(Self {
            device,
            file_header,
            workgroup_size,
            mv_mul_bind_group_layout,
            copy_bind_group_layout,
            globals_alignment,
            copy_dim_alignment,
            globals,
            copy_dims,
            vector_a_buffer,
            vector_b_buffer,
            output_vector_buffer,
            download_buffer,
            copy_pipeline,
            mv_mul_pipeline,
        })
    }

    fn bind_matrices(&self, matrices: &[u8]) -> Result<Runner<'_>> {
        let globals_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("globals_buffer"),
                contents: &self.globals,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // TODO: is COPY_DST necessary?
            });

        let copy_dims_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("copy_dims_buffer"),
                contents: &self.copy_dims,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // TODO: is COPY_DST necessary?
            });

        let matrices_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("matrices_buffer"),
                contents: matrices,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mv_mul_even_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("even"),
            layout: &self.mv_mul_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(self.globals_alignment as u64).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrices_buffer.as_entire_binding(), // TODO: try `.slice()` instead.
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.vector_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vector_b_buffer.as_entire_binding(),
                },
            ],
        });

        let mv_mul_odd_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("odd"),
            layout: &self.mv_mul_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(self.globals_alignment as u64).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrices_buffer.as_entire_binding(), // TODO: try `.slice()` instead.
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.vector_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.vector_a_buffer.as_entire_binding(),
                },
            ],
        });

        let copy_out_even_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &copy_dims_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(self.copy_dim_alignment as u64).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.vector_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.output_vector_buffer.as_entire_binding(),
                },
            ],
        });

        let copy_out_odd_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &copy_dims_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(self.copy_dim_alignment as u64).unwrap()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.vector_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.output_vector_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Runner {
            ops: self,
            mv_mul_bind_groups: [mv_mul_even_bind_group, mv_mul_odd_bind_group],
            copy_out_bind_groups: [copy_out_even_bind_group, copy_out_odd_bind_group],
            copy_dims_buffer,
        })
    }
}

struct Runner<'ops> {
    ops: &'ops MatVecMulOps<'ops, 'ops>,
    mv_mul_bind_groups: [wgpu::BindGroup; 2],
    copy_out_bind_groups: [wgpu::BindGroup; 2],
    copy_dims_buffer: wgpu::Buffer,
}

impl Runner<'_> {
    fn run(
        &self,
        queue: &wgpu::Queue,
        input_vector: &[i8],
        matrix_ids: std::ops::Range<u32>,
    ) -> Result<DownloadVecGuard<'_>> {
        let input_vector_buffer =
            self.ops
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("input_vector_buffer"),
                    contents: bytemuck::cast_slice(input_vector),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let copy_in_bind_group = self
            .ops
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("copy_in"),
                layout: &self.ops.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.copy_dims_buffer,
                            offset: 0,
                            size: Some(
                                NonZeroU64::new(self.ops.copy_dim_alignment as u64).unwrap(),
                            ),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_vector_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.ops.vector_a_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ops
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut compute_pass_copy_in = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass_copy_in.set_pipeline(&self.ops.copy_pipeline);
        compute_pass_copy_in.set_bind_group(0, &copy_in_bind_group, &[0]);
        let workgroup_count = self.ops.file_header.dimensions[matrix_ids.start as usize]
            .div_ceil(self.ops.workgroup_size);
        compute_pass_copy_in.dispatch_workgroups(workgroup_count, 1, 1);
        drop(compute_pass_copy_in);

        for (k, bind_group) in matrix_ids
            .clone()
            .zip(self.mv_mul_bind_groups.iter().cycle())
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.ops.mv_mul_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[k * self.ops.globals_alignment]);
            let workgroup_count =
                self.ops.file_header.dimensions[k as usize + 1].div_ceil(self.ops.workgroup_size);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy the result to a buffer that is marked for `COPY_SRC` usage.
        let mut compute_pass_copy_out = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass_copy_out.set_pipeline(&self.ops.copy_pipeline);
        compute_pass_copy_out.set_bind_group(
            0,
            &self.copy_out_bind_groups[matrix_ids.len() % 2],
            &[self.ops.copy_dim_alignment],
        );
        let workgroup_count = self.ops.file_header.dimensions[matrix_ids.end as usize]
            .div_ceil(self.ops.workgroup_size);
        compute_pass_copy_out.dispatch_workgroups(workgroup_count, 1, 1);
        drop(compute_pass_copy_out);

        // Download the result from the GPU to the CPU.
        encoder.copy_buffer_to_buffer(
            &self.ops.output_vector_buffer,
            0,
            &self.ops.download_buffer,
            0,
            self.ops.file_header.dimensions[matrix_ids.end as usize] as u64,
        );

        let command_buffer = encoder.finish();

        queue.submit([command_buffer]);

        // We now map the download buffer so we can read it. Mapping tells wgpu that we want to
        // read/write to the buffer directly by the CPU and it should not permit any more GPU
        // operations on the buffer. Mapping requires that the GPU be finished using the buffer
        // before it resolves, so mapping has a callback to tell you when the mapping is complete.
        DownloadVecGuard::new(self.ops.device, &self.ops.download_buffer)
    }
}

struct DownloadVecGuard<'buf> {
    buf: &'buf wgpu::Buffer,
    buf_view: std::mem::ManuallyDrop<wgpu::BufferView<'buf>>,
}

impl Drop for DownloadVecGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: We don't use the `buf_view` anymore after manually dropping it. Further, we're in
        // the destructor, and we own the `buf_view`, so no one else can use it afterwards either.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.buf_view) };
        self.buf.unmap();
    }
}

impl Deref for DownloadVecGuard<'_> {
    type Target = [i8];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.buf_view)
    }
}

impl<'buf> DownloadVecGuard<'buf> {
    fn new(device: &wgpu::Device, buf: &'buf wgpu::Buffer) -> Result<Self> {
        let buf_slice = buf.slice(..);
        let result = Arc::new(Mutex::new(Ok(())));
        let inner_result = Arc::clone(&result);
        buf_slice.map_async(wgpu::MapMode::Read, move |r| {
            // On native, `self.device.poll()` below blocks until mapping is finished, so this
            // callback is mostly for error handling.
            *inner_result.lock().unwrap() = r;
        });

        // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU,
        // so we would need to rely on the callback to know when the buffer is mapped.
        device.poll(wgpu::PollType::Wait)?;
        std::mem::replace(&mut *result.lock().unwrap(), Ok(()))?; // Propagate any errors.

        // We can now read the data from the buffer.
        let buf_view = buf_slice.get_mapped_range();
        Ok(Self {
            buf,
            buf_view: std::mem::ManuallyDrop::new(buf_view),
        })
    }
}
