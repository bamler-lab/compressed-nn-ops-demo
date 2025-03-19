use std::{io::Seek, path::PathBuf};

use anyhow::{Result, bail};
use clap::Parser;

use compressed_mat_vec_mul::{
    CompressedMatrix, FileHeader, RngSeeder, UncompressedMatrix, UncompressedVector,
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

    /// Standard deviation of the Gaussian distribution used to generate the random values for both
    /// the input vector and the matrices.
    #[arg(short, long, default_value = "4.0")]
    std: f64,

    /// The seed to use for the random number generator. If not provided, a hash of the matrix
    /// `num_matrices` and `dim` will be used as a seed to ensure reproducibility.
    #[arg(long, default_value = "20250319")]
    seed: u64,

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

        let compressed = CompressedMatrix::from_read(&mut reader, cli.dim)?;
        let uncompressed = compressed.to_uncompressed(cli.dim);

        assert!(uncompressed == uncompressed_ground_truth); // Don't use `assert_eq!` because matrices are too big to print.
    }

    progress_bar.finish();

    Ok(())
}
