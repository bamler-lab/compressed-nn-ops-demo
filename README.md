# GPU Kernels for Compressed Neural Networks

<hr>

⚠️ **CAUTION:**
This repository contains a portable proof of concept to demonstrate a new idea for speeding up LLM inference, *implemented in WebGPU Shading Language (WGSL)*.
**We are currently working on a more serious implementation in CUDA (based on [CUTLASS](https://github.com/NVIDIA/cutlass)) for NVIDIA-GPUs, which we will publish alongside a corresponding paper once it's ready.**

<hr>

This repository contains some proof-of-concept implementations of GPU kernels that are intended to speed up matrix-vector multiplications by operating directly on compressed (i.e., [entropy coded](https://en.wikipedia.org/wiki/Entropy_coding)) matrices.
My goal of this repository is to demonstrate that real model *compression* (not just model quantization) can help speeding up inference by reducing the amount of required memory access in bandwidth-bound operations (which are usually the bottleneck in LLM inference).

For more on the difference between model compression and model quantization, see Section 3.1 of [our recent paper [1]](https://arxiv.org/pdf/2505.18758).

## Scope and Directory Outline

**The most relevant demo is in the directory  [`wgpu`](./wgpu).**
It contains a WebGPU (WGSL) implementation of a GPU kernel (a so-called compute shader) for matrix-vector multiplication, which is the main operation performed during inference in large language models (LLMs).
The GPU kernel operates directly on a compressed representation of the matrices, and it decompresses each matrix element on the fly to perform the matrix-vector multiplication.
Despite the additional computation required for decompression, our kernel turns out to be faster than a simple baseline that operates on uncompressed matrices (because the compressed representation of the matrix is smaller, so operating on it requires fewer memory accesses).

**Caution:**
please take these preliminary results with a grain of salt.
Neither the GPU kernel for compressed matrix-vector multiplication nor the baseline for uncompressed matrices is particularly well optimized yet.
However, even if these preliminary results overestimate the speedups, portable GPU kernels that operate directly on compressed matrices are still extremely useful for deep learning on edge devices because they allow us to fit larger models on GPUs with limited memory.
From this perspective, it would already be an achievement to operate on compressed matrices without slowing down inference, let alone speeding it up.

## Technical Details and Distinction From Prior Work

For most of the operations during inference in large language models (LLMs), the bottleneck is not computation but _memory access_ [[2](https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf)].
This is because, once the prompt has been parsed and the first output token has been generated, subsequent tokens are generated autoregressively (i.e., generating one token at a time), and most of the time is spent on matrix-vector multiplications.
Matrix-vector multiplications are highly _bandwidth bound_ because each neural network weight is used in only a single multiply-accumulate (MAC) operation (in contrast to LLM training and prompt parsing, which involve more matrix-matrix multiplications and are often compute-bound).

**Existing approaches** to address the memory bottleneck in LLM inference, especially on edge devices, involve:

1. weight quantization and
2. speculative execution.

The method demonstrated in this repository is related to weight quantization but takes the idea further by combining it with modern techniques for lossless data compression (aka entropy coding).
Rather than just rounding each neural network weight to a fixed precision (e.g., 4 bit per weight), as is done in weight quantization, the demos in this repository load the matrices on the GPU in compressed form (think "gzip" but faster and more effective, see [[1]](https://arxiv.org/pdf/2505.18758) for details).
We decompress the matrix entries on-the-fly directly on the GPU during the matrix-vector multiplication.
Since each decompressed matrix entry is used in only a single MAC operation, we can then immediately throw it away, and we never have to hold the entire decompressed matrix in memory.

Despite the additional computation that is necessary for decompression, preliminary results indicate that our approach ends up being faster than performing the same matrix-vector multiplications in uncompressed form.
This can be explained by the fact that entropy coding compresses the matrices to a shorter bit string, thus reducing the number of memory accesses by the GPU.
Also, modern entropy coders can decompress data extremely fast when using adequate entropy models (our demos currently use an ANS entropy coder [[3](https://ieeexplore.ieee.org/abstract/document/7170048), [4](https://arxiv.org/pdf/2201.01741)]).


## References

1. Conzelmann and Bamler, "Reducing Storage of Pretrained Neural Networks by Rate-Constrained Quantization and Entropy Coding." arXiv preprint, [arXiv:2505.18758 (2025)](https://arxiv.org/pdf/2505.18758).
2. Ivanov, et al., "Data movement is all you need: A case study on optimizing transformers." [Proceedings of Machine Learning and Systems 3 (2021): 711-732](https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf).
3. Duda, et al., "The use of asymmetric numeral systems as an accurate replacement for Huffman coding." [2015 Picture Coding Symposium (PCS). IEEE, 2015](https://ieeexplore.ieee.org/abstract/document/7170048).
4. Bamler, "Understanding entropy coding with asymmetric numeral systems (ANS): a statistician's perspective." arXiv preprint, [arXiv:2201.01741 (2022)](https://arxiv.org/pdf/2201.01741).
