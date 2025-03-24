use std::{io::Seek, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use log::info;
use rand::RngCore;

use compressed_mat_vec_mul::{
    CompressedMatrix, FileHeader, RngSeeder, UncompressedMatrix, UncompressedVector,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to output file. If the file already exists it will be overwritten. Otherwise, a new
    /// file will be created.
    output: PathBuf,

    /// The number of matrices to generate.
    #[arg(short = 'k', long, default_value = "100")]
    num_matrices: u32,

    /// The number of rows and columns in each matrix, and the dimension of the input vector.
    #[arg(short, long, default_value = "4096")]
    dim: u32,

    /// Standard deviation of the Gaussian distribution used to generate the random values for both
    /// the input vector and the matrices.
    #[arg(short, long, default_value = "4.0")]
    std: f64,

    /// Write the matrices in uncompressed form. This is meant for baseline performance testing.
    #[arg(long)]
    uncompressed: bool,

    /// The seed to use for the random number generator. If not provided, a hash of the matrix
    /// `num_matrices` and `dim` will be used as a seed to ensure reproducibility. Set to zero to
    /// use a random seed.
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
        let seed = rand::rng().next_u64();
        info!("Using randomly generated seed {}.", seed);
        seed
    } else {
        cli.seed
    };

    let rng_seeder = RngSeeder::new((seed, cli.num_matrices, cli.dim));
    let distribution = probability::distribution::Gaussian::new(0.0, cli.std);

    let file = std::fs::File::create(&cli.output)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut cursor = 0;

    let mut file_header = FileHeader::stump(cli.num_matrices, cli.dim);
    cursor += file_header.to_write(&mut writer)?;

    let input_vector =
        UncompressedVector::random(cli.dim, distribution, rng_seeder.rng("input_vector"))?;
    cursor += input_vector.to_write(&mut writer)?;

    let mut intermediate_vector = input_vector.clone();

    let progress_bar = indicatif::ProgressBar::new(cli.num_matrices as u64);
    progress_bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{bar:40} {pos}/{len} [{elapsed_precise}]")
            .expect("valid template"),
    );

    for k in 0..cli.num_matrices {
        progress_bar.set_position(k as u64);
        file_header.offsets_and_file_size[k as usize] = cursor;

        let uncompressed = UncompressedMatrix::random(
            cli.dim,
            cli.dim,
            distribution,
            &mut intermediate_vector,
            rng_seeder.rng(("matrix", k)),
        )?;

        let bytes_written = if cli.uncompressed {
            uncompressed.to_write(&mut writer)?
        } else {
            let compressed = CompressedMatrix::from_uncompressed(&uncompressed);
            compressed.to_write(&mut writer)?
        };

        cursor += bytes_written;
        if bytes_written as u32 > file_header.max_compressed_matrix_size {
            file_header.max_compressed_matrix_size = bytes_written as u32;
        }
    }

    file_header.offsets_and_file_size[cli.num_matrices as usize] = cursor;

    writer.seek(std::io::SeekFrom::Start(0))?;
    file_header.to_write(&mut writer)?;

    progress_bar.finish();

    Ok(())
}
