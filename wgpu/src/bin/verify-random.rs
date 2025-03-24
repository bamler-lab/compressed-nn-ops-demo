use std::{collections::HashMap, io::Seek, path::PathBuf};

use anyhow::{Result, bail};
use clap::Parser;

use compressed_mat_vec_mul::{
    CompressedMatrix, FileHeader, OwnedSafetensor, RngSeeder, UncompressedMatrix,
    UncompressedVector,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to input file (use `mk-random` to create a suitable file).
    input: PathBuf,

    /// The number of matrices to expect.
    #[arg(short = 'k', long, default_value = "100")]
    num_matrices: u32,

    /// The number of rows and columns to expect in each matrix, and the dimension to expect for the
    /// input vector.
    #[arg(short, long, default_value = "4096")]
    dim: u32,

    #[arg(long)]
    subgroup_size: u32,

    /// Standard deviation of the Gaussian distribution used to generate the random values for both
    /// the input vector and the matrices.
    #[arg(short, long, default_value = "4.0")]
    std: f64,

    /// The seed to use for the random number generator. If not provided, a hash of the matrix
    /// `num_matrices` and `dim` will be used as a seed to ensure reproducibility.
    #[arg(long, default_value = "20250319")]
    seed: u64,

    /// Write the input vector, all intermediate vectors, and the final result to the provided file
    /// in `safetensors` format. If the file already exists it will be overwritten. Otherwise, a new
    /// file will be created.
    #[arg(long)]
    debug: Option<PathBuf>,

    /// Include the (uncompressed) matrices in the debug file (requires `--debug`). This is turned
    /// off by default because it can lead to a fairly large debug file.
    #[arg(long, requires = "debug")]
    debug_matrices: bool,

    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // wgpu uses `log` rather than `tracing`, so we'll use that too.
    env_logger::Builder::new()
        .filter_level(cli.verbose.log_level_filter())
        .init();

    let seed = if cli.seed == 0 {
        bail!(
            "The seed must not be zero. If `mk-random` was run with `--seed 0` then it used\n\
            a randomly generated seed. You can get the actual seed by further adding the\n\
            switches `-vv` to the `mk-random` call."
        );
    } else {
        cli.seed
    };

    let rng_seeder = RngSeeder::new((seed, cli.num_matrices, cli.dim));
    let distribution = probability::distribution::Gaussian::new(0.0, cli.std);

    let file = std::fs::File::open(&cli.input)?;
    let mut reader = std::io::BufReader::new(file);

    let file_header = FileHeader::from_read(&mut reader)?;
    assert_eq!(file_header.num_matrices(), cli.num_matrices);
    assert_eq!(
        file_header.dimensions,
        vec![cli.dim; cli.num_matrices as usize + 1].into_boxed_slice()
    );

    let input_vector_ground_truth =
        UncompressedVector::random(cli.dim, distribution, rng_seeder.rng("input_vector"))?;
    let input_vector = UncompressedVector::from_read(&mut reader, cli.dim)?;
    assert!(input_vector == input_vector_ground_truth); // Don't use `assert_eq!` because vectors are too big to print.

    let mut intermediate_vector = input_vector.clone();

    let progress_bar = indicatif::ProgressBar::new(cli.num_matrices as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} [{elapsed_precise}]")
            .expect("valid template"),
    );

    let mut debug_data = cli.debug.map(|debug_path| {
        let mut debug_data = HashMap::with_capacity(cli.num_matrices as usize + 1);
        debug_data.insert("vector_0".to_string(), input_vector.into_owned_safetensor());
        (debug_path, debug_data)
    });

    for k in 0..cli.num_matrices {
        progress_bar.set_position(k as u64);
        reader.seek(std::io::SeekFrom::Start(
            file_header.offsets_and_file_size[k as usize] as u64,
        ))?;

        let uncompressed_ground_truth = UncompressedMatrix::random(
            cli.dim,
            cli.dim,
            distribution,
            &mut intermediate_vector,
            rng_seeder.rng(("matrix", k)),
        )?;

        let compressed = CompressedMatrix::from_read(&mut reader, cli.dim, cli.subgroup_size)?;
        let uncompressed = compressed.to_uncompressed(cli.dim, cli.subgroup_size);

        assert!(uncompressed == uncompressed_ground_truth); // Don't use `assert_eq!` because matrices are too big to print.

        if let Some((_, debug_data)) = &mut debug_data {
            if cli.debug_matrices {
                debug_data.insert(
                    format!("grid_spacing_{}", k),
                    OwnedSafetensor::from_scalar_f32(uncompressed.grid_spacing().to_f32()),
                );
                debug_data.insert(
                    format!("matrix_{}", k),
                    uncompressed.into_owned_safetensor(),
                );
            }
            debug_data.insert(
                format!("vector_{}", k + 1),
                intermediate_vector.clone().into_owned_safetensor(),
            );
        }
    }

    if let Some((debug_path, debug_data)) = debug_data {
        safetensors::serialize_to_file(debug_data, &None, &debug_path)?;
    }

    progress_bar.finish();

    Ok(())
}
