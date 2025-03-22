use std::{
    hash::{Hash, Hasher},
    io::{Read, Write},
    num::NonZeroU8,
};

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use constriction::stream::model::{ContiguousCategoricalEntropyModel, IterableEntropyModel};
use probability::{distribution::Sample, source::Source};
use rayon::prelude::*;

mod f16;
pub use f16::SimpleF16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileHeader {
    /// Measured in bytes.
    pub max_compressed_matrix_size: u32,

    /// Has length `num_matrices + 1`. Starts with the dimension of the input vector, followed by
    /// the row dimensions of each matrix.
    pub dimensions: Box<[u32]>,

    /// Measured in bytes, from the beginning of the file.
    pub offsets_and_file_size: Box<[u32]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UncompressedVector(pub Vec<i8>);

#[derive(Clone, Debug, PartialEq)]
pub struct UncompressedMatrix {
    rows: u32,
    cols: u32,
    grid_spacing: SimpleF16,
    data: Box<[i8]>,
}

#[derive(Clone, Debug)]
pub struct CompressedMatrix {
    grid_spacing: SimpleF16,
    grid_start: i8,

    /// Includes a final entry that's always zero. Does not include padding.
    cdf: Box<[u8]>,

    /// Measured in units of 4-bytes, from the start of `compressed_data`; includes a final entry
    /// that points just past the end of `compressed_data`
    offsets: Box<[u32]>,
    compressed_data: Box<[u32]>,
}

#[derive(Clone, Debug)]
pub struct OwnedSafetensor {
    dtype: safetensors::Dtype,
    shape: Box<[usize]>,
    data: Box<[u8]>,
    data_len: usize,
}

impl OwnedSafetensor {
    pub fn from_scalar_f32(scalar: f32) -> Self {
        let data = Box::new(scalar.to_le_bytes());
        OwnedSafetensor {
            dtype: safetensors::Dtype::F32,
            shape: vec![].into_boxed_slice(),
            data,
            data_len: std::mem::size_of::<f32>(),
        }
    }
}

impl safetensors::View for OwnedSafetensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        bytemuck::cast_slice(&self.data).into()
    }

    fn data_len(&self) -> usize {
        self.data_len
    }
}

impl FileHeader {
    pub fn stump(num_matrices: u32, constant_dimension: u32) -> Self {
        Self {
            max_compressed_matrix_size: 0,
            dimensions: vec![constant_dimension; num_matrices as usize + 1].into_boxed_slice(),
            offsets_and_file_size: vec![0; num_matrices as usize + 1].into_boxed_slice(),
        }
    }

    pub fn num_matrices(&self) -> u32 {
        self.dimensions.len() as u32 - 1
    }

    pub fn from_read(mut reader: impl Read) -> Result<Self> {
        let num_matrices = reader.read_u32::<LittleEndian>()?;
        let max_compressed_matrix_size = reader.read_u32::<LittleEndian>()?;
        let dimensions = (0..num_matrices + 1)
            .map(|_| reader.read_u32::<LittleEndian>())
            .collect::<Result<Box<[u32]>, _>>()?;
        let offsets_and_file_size = (0..(num_matrices + 1))
            .map(|_| reader.read_u32::<LittleEndian>())
            .collect::<Result<Box<[u32]>, _>>()?;

        Ok(Self {
            max_compressed_matrix_size,
            dimensions,
            offsets_and_file_size,
        })
    }

    /// Returns the number of bytes written.
    pub fn to_write(&self, mut writer: impl Write) -> Result<u32> {
        writer.write_u32::<LittleEndian>(self.num_matrices())?;
        writer.write_u32::<LittleEndian>(self.max_compressed_matrix_size)?;
        for &dim in &self.dimensions {
            writer.write_u32::<LittleEndian>(dim)?;
        }
        for &offset in &self.offsets_and_file_size {
            writer.write_u32::<LittleEndian>(offset)?;
        }

        Ok(4 * (2 + self.dimensions.len() + self.offsets_and_file_size.len()) as u32)
    }
}

impl UncompressedVector {
    pub fn random(
        dim: u32,
        distribution: impl Sample<Value = f64>,
        mut rng: impl Source,
    ) -> Result<Self> {
        let vector = (0..dim)
            .map(|_| i8::try_from(distribution.sample(&mut rng).round() as i32))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| "Generated value out of bounds")?;
        Ok(Self(vector))
    }

    pub fn from_read(mut reader: impl Read, dimension: u32) -> Result<Self> {
        let mut vector = vec![0i8; dimension as usize];
        reader.read_exact(bytemuck::cast_slice_mut(&mut vector))?;

        let num_padding = 3 - (vector.len() + 3) % 4;
        reader.read_exact(&mut [0u8; 3][0..num_padding])?; // Ignore the padding.

        Ok(Self(vector))
    }

    /// Returns the number of bytes written.
    pub fn to_write(&self, mut writer: impl Write) -> Result<u32> {
        writer.write_all(bytemuck::cast_slice(&self.0))?;

        // Pad to a multiple of 4 bytes.
        let num_padding = 3 - (self.0.len() + 3) % 4;
        writer.write_all(&[0u8; 3][0..num_padding])?;

        Ok((self.0.len() + num_padding) as u32)
    }

    pub fn into_owned_safetensor(self) -> OwnedSafetensor {
        let len = self.0.len();
        OwnedSafetensor {
            dtype: safetensors::Dtype::I8,
            shape: vec![len].into_boxed_slice(),
            data: bytemuck::allocation::cast_vec(self.0).into_boxed_slice(),
            data_len: len,
        }
    }
}

impl UncompressedMatrix {
    pub fn random(
        rows: u32,
        cols: u32,
        distribution: impl Sample<Value = f64>,
        intermediate_vector: &mut UncompressedVector,
        mut rng: impl Source,
    ) -> Result<Self> {
        assert_eq!(intermediate_vector.0.len(), cols as usize);

        let data = (0..rows * cols)
            .map(|_| i8::try_from(distribution.sample(&mut rng).round() as i32))
            .collect::<Result<Box<[i8]>, _>>()
            .with_context(|| "Generated value out of bounds")?;

        // Find a reasonable grid spacing.
        let output_vec = (0..rows)
            .into_par_iter()
            .map(|row| {
                let start = (row * cols) as usize;
                let end = start + cols as usize;
                data[start..end]
                    .iter()
                    .zip(&intermediate_vector.0)
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum::<i32>()
            })
            .collect::<Vec<_>>();

        let output_max = output_vec
            .iter()
            .map(|x| x.abs())
            .max()
            .expect("rows should be > 0");
        let grid_spacing = SimpleF16::from_f32(127.0 / output_max as f32);
        let grid_spacing_f32 = grid_spacing.to_f32();

        intermediate_vector.0.resize(rows as usize, 0);
        for (&src, dst) in output_vec.iter().zip(&mut intermediate_vector.0) {
            *dst = (grid_spacing_f32 * src as f32).round_ties_even() as i8;
        }

        Ok(Self {
            rows,
            cols,
            grid_spacing,
            data,
        })
    }

    pub fn into_owned_safetensor(self) -> OwnedSafetensor {
        let len = self.data.len();
        OwnedSafetensor {
            dtype: safetensors::Dtype::I8,
            shape: vec![self.rows as usize, self.cols as usize].into_boxed_slice(),
            data: bytemuck::allocation::cast_slice_box(self.data),
            data_len: len,
        }
    }

    pub fn grid_spacing(&self) -> SimpleF16 {
        self.grid_spacing
    }
}

impl CompressedMatrix {
    pub fn from_uncompressed(uncompressed: &UncompressedMatrix) -> Self {
        assert_eq!(uncompressed.cols % 4, 0);

        let mut counts = [0u32; 256];
        for &x in &uncompressed.data {
            counts[(x as i32 + 128) as usize] += 1;
        }
        let grid_start_index = counts
            .iter()
            .position(|&x| x > 0)
            .expect("non-empty matrix");
        let grid_start = (grid_start_index as i32 - 128) as i8;
        let grid_end_index = counts
            .iter()
            .rposition(|&x| x > 0)
            .expect("non-empty matrix")
            + 1;

        let unnormalized_probabilities = counts[grid_start_index..grid_end_index]
            .iter()
            .map(|&x| x as f32)
            .collect::<Vec<_>>();

        let entropy_model =
            ContiguousCategoricalEntropyModel::<u8, Vec<u8>,8>::from_floating_point_probabilities_perfect(
                &unnormalized_probabilities,
            )
            .expect("must be normalizable");
        let cdf = entropy_model
            .symbol_table()
            .map(|(_symbol, left_cdf, _probability)| left_cdf)
            .chain(std::iter::once(0))
            .collect::<Box<[u8]>>();

        let mut compressed_data = Vec::new();
        let mut offsets = Vec::with_capacity(uncompressed.rows as usize + 1);

        for row in (0..uncompressed.rows).rev() {
            offsets.push(compressed_data.len() as u32);

            // TODO: is this reset really necessary after each row?
            let mut state = 1u64 << 32;

            let start = (row * uncompressed.cols) as usize;
            let end = start + uncompressed.cols as usize;
            let chunks: &[[i8; 4]] = bytemuck::cast_slice(&uncompressed.data[start..end]);
            for chunk in chunks.iter().rev() {
                let index0 = (chunk[0] as i32 - grid_start as i32) as usize;
                let left_cdf0 = cdf[index0];
                let right_cdf0 = cdf[index0 + 1];
                let probability0 = right_cdf0.wrapping_sub(left_cdf0) as u32;

                let index1 = (chunk[1] as i32 - grid_start as i32) as usize;
                let left_cdf1 = cdf[index1];
                let right_cdf1 = cdf[index1 + 1];
                let probability1 = right_cdf1.wrapping_sub(left_cdf1) as u32;

                let index2 = (chunk[2] as i32 - grid_start as i32) as usize;
                let left_cdf2 = cdf[index2];
                let right_cdf2 = cdf[index2 + 1];
                let probability2 = right_cdf2.wrapping_sub(left_cdf2) as u32;

                let index3 = (chunk[3] as i32 - grid_start as i32) as usize;
                let left_cdf3 = cdf[index3];
                let right_cdf3 = cdf[index3 + 1];
                let probability3 = right_cdf3.wrapping_sub(left_cdf3) as u32;

                if (state >> 32) as u32 >= probability0 * probability1 * probability2 * probability3
                {
                    // The planned updates would overflow the state, so we need to flush a part of it.
                    compressed_data.push((state & ((1u64 << 32) - 1)) as u32);
                    state >>= 32;
                    // At this point, `state >> 32 == 0`.
                } // If the branch was not taken, then `state >> 32 != 0` at this point.

                let remainder3 = (state % probability3 as u64) as u32;
                state /= probability3 as u64;
                let mut lower_state = left_cdf3 as u32 + remainder3;

                let remainder2 = (state % probability2 as u64) as u32;
                state /= probability2 as u64;
                lower_state <<= 8;
                lower_state |= left_cdf2 as u32 + remainder2;

                let remainder1 = (state % probability1 as u64) as u32;
                state /= probability1 as u64;
                lower_state <<= 8;
                lower_state |= left_cdf1 as u32 + remainder1;

                let remainder0 = (state % probability0 as u64) as u32;
                state /= probability0 as u64;
                lower_state <<= 8;
                lower_state |= left_cdf0 as u32 + remainder0;

                assert_eq!(state >> 32, 0);

                state = (state << 32) | lower_state as u64;
            }

            compressed_data.push((state & ((1u64 << 32) - 1)) as u32);
            compressed_data.push((state >> 32) as u32);
        }

        offsets.push(compressed_data.len() as u32);
        for offset in &mut offsets {
            *offset = compressed_data.len() as u32 - *offset;
        }

        compressed_data.reverse();
        offsets.reverse();

        Self {
            grid_spacing: uncompressed.grid_spacing,
            grid_start,
            cdf,
            offsets: offsets.into_boxed_slice(),
            compressed_data: compressed_data.into_boxed_slice(),
        }
    }

    pub fn to_uncompressed(&self, cols: u32) -> UncompressedMatrix {
        assert_eq!(cols % 4, 0);

        let rows = self.offsets.len() as u32 - 1;

        #[derive(Debug, Clone, Copy)]
        struct PpfEntry {
            left_cdf: u8,
            probability: NonZeroU8,
            symbol: i8,
        }

        let mut ppf = Vec::with_capacity(256);

        for ((symbol, &left_cdf), &right_cdf) in (self.grid_start..)
            .zip(&self.cdf)
            .zip(self.cdf.iter().skip(1))
        {
            if let Some(probability) = NonZeroU8::new(right_cdf.wrapping_sub(left_cdf)) {
                for _ in 0..probability.get() {
                    ppf.push(PpfEntry {
                        left_cdf,
                        probability,
                        symbol,
                    });
                }
            }
        }

        assert_eq!(ppf.len(), 256);

        let mut uncompressed_data = Vec::with_capacity((rows * cols) as usize);

        for row in 0..rows {
            let mut cursor = self.offsets[row as usize];

            let mut state = self.compressed_data[cursor as usize] as u64;
            state <<= 32;
            cursor += 1;
            state |= self.compressed_data[cursor as usize] as u64;
            cursor += 1;

            for _ in 0..cols / 4 {
                let lower_state = (state & ((1u64 << 32) - 1)) as u32;
                state >>= 32;

                let quantile0 = (lower_state & 0xff) as u8;
                let quantile1 = ((lower_state >> 8) & 0xff) as u8;
                let quantile2 = ((lower_state >> 16) & 0xff) as u8;
                let quantile3 = ((lower_state >> 24) & 0xff) as u8;

                let ppf_entry0 = ppf[quantile0 as usize];
                let ppf_entry1 = ppf[quantile1 as usize];
                let ppf_entry2 = ppf[quantile2 as usize];
                let ppf_entry3 = ppf[quantile3 as usize];

                uncompressed_data.push(ppf_entry0.symbol);
                uncompressed_data.push(ppf_entry1.symbol);
                uncompressed_data.push(ppf_entry2.symbol);
                uncompressed_data.push(ppf_entry3.symbol);

                let remainder0 = quantile0 - ppf_entry0.left_cdf;
                let remainder1 = quantile1 - ppf_entry1.left_cdf;
                let remainder2 = quantile2 - ppf_entry2.left_cdf;
                let remainder3 = quantile3 - ppf_entry3.left_cdf;

                let mut full_remainder = remainder0 as u32 * ppf_entry1.probability.get() as u32;
                full_remainder += remainder1 as u32;
                full_remainder *= ppf_entry2.probability.get() as u32;
                full_remainder += remainder2 as u32;
                full_remainder *= ppf_entry3.probability.get() as u32;
                full_remainder += remainder3 as u32;

                let full_probability = ppf_entry0.probability.get() as u32
                    * ppf_entry1.probability.get() as u32
                    * ppf_entry2.probability.get() as u32
                    * ppf_entry3.probability.get() as u32;

                state = full_probability as u64 * state + full_remainder as u64;

                if state >> 32 == 0 {
                    // Refill the state as soon as we can.
                    state = (state << 32) | self.compressed_data[cursor as usize] as u64;
                    cursor += 1;
                }
            }

            assert_eq!(state, 1u64 << 32);
            assert_eq!(cursor, self.offsets[(row + 1) as usize]);
        }

        UncompressedMatrix {
            rows,
            cols,
            grid_spacing: self.grid_spacing,
            data: uncompressed_data.into_boxed_slice(),
        }
    }

    /// Returns the number of bytes written.
    pub fn to_write(&self, mut writer: impl Write) -> Result<u32> {
        writer.write_u16::<LittleEndian>(self.grid_spacing.to_bits())?;
        writer.write_i8(self.grid_start)?;
        let grid_size = (self.cdf.len() - 1) as u8;
        writer.write_u8(grid_size)?;

        writer.write_all(&self.cdf)?;
        // Pad to a multiple of 4 bytes.
        let num_padding = 3 - (self.cdf.len() + 3) % 4;
        writer.write_all(&[0u8; 3][0..num_padding])?;

        for &offset in &self.offsets {
            writer.write_u32::<LittleEndian>(offset)?;
        }
        for &data in &self.compressed_data {
            writer.write_u32::<LittleEndian>(data)?;
        }

        Ok((4
            + self.cdf.len()
            + num_padding
            + 4 * (self.offsets.len() + self.compressed_data.len())) as u32)
    }

    pub fn from_read(mut reader: impl Read, rows: u32) -> Result<Self> {
        let grid_spacing = SimpleF16::from_bits(reader.read_u16::<LittleEndian>()?);
        let grid_start = reader.read_i8()?;
        let grid_size = reader.read_u8()?;

        let mut cdf = vec![0u8; grid_size as usize + 1];
        reader.read_exact(&mut cdf)?;
        // Ignore the padding.
        let num_padding = 3 - (cdf.len() + 3) % 4;
        reader.read_exact(&mut [0u8; 3][0..num_padding])?;

        let offsets = (0..rows + 1)
            .map(|_| reader.read_u32::<LittleEndian>())
            .collect::<Result<Box<[_]>, _>>()?;
        let compressed_data = (0..offsets[rows as usize])
            .map(|_| reader.read_u32::<LittleEndian>())
            .collect::<Result<Box<[_]>, _>>()?;

        Ok(Self {
            grid_spacing,
            grid_start,
            cdf: cdf.into_boxed_slice(),
            offsets,
            compressed_data,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RngSeeder {
    hasher: fxhash::FxHasher32,
}

impl RngSeeder {
    pub fn new(global_seed: impl Hash) -> Self {
        let mut hasher = fxhash::FxHasher32::default();
        global_seed.hash(&mut hasher);
        Self { hasher }
    }

    pub fn rng(&self, id: impl Hash) -> probability::source::Xorshift128Plus {
        let mut hasher = self.hasher.clone();
        id.hash(&mut hasher);
        let seed1 = hasher.finish();
        hasher.write_u32(0);
        let seed2 = hasher.finish();
        probability::source::Xorshift128Plus::new([seed1, seed2])
    }
}
