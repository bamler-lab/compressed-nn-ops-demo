# GPU Shaders for Compressed Neural Networks

This repository contains some proof-of-concept implementations of GPU shaders for consumer-grade GPUs that are intended to speed up matrix-vector multiplications by operating directly on compressed (i.e., entropy coded) matrices.
My goal of this repository is to demonstrate that model compression can make it easier to deploy transformer-based language models (LLMs and SLMs) to consumer devices (i.e., laptops or mobile devices) by

1. reducing the amount of data that has to fit on a GPU without sacrificing the size of the neural network; and
2. speeding up inference by reducing the number of memory accesses required in bandwidth-bound operations (which are usually the bottleneck in LLM inference).

## Scope and Directory Outline

**The most relevant demo is in the directory  [`wgpu`](./wgpu).**

This repository is a **proof of concept**.
The directory [`wgpu`](./wgpu) contains a WebGPU (WGSL) implementation of a GPU kernel (a so-called compute shader) for matrix-vector multiplication, which is the main operation performed during inference in large language models.
The GPU kernel operates directly on a compressed representation of the matrices, and despite the additional computation required for decompression, it turns out to be faster than a simple baseline that operates on uncompressed matrices (because the compressed representation is smaller, so operating on it requires fewer memory accesses).

**Caution:**
please take these preliminary results with a grain of salt.
Neither the GPU kernel for compressed matrix-vector multiplication nor the baseline for uncompressed matrices is particularly well optimized yet.
However, even if these preliminary results overestimate the speedups, portable GPU kernels that operate directly on compressed matrices are still extremely useful for deep learning on edge devices because they allow us to fit larger models on GPUs with limited memory.
From this perspective, it would already be an achievement to operate on compressed matrices without slowing down inference, let alone speeding it up.

## Technical Details and Distinction From Prior Work

For most of the operations during inference in large language models (LLMs), the bottleneck is not computation but _memory access_ [[1](https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf)].
This is because, once the prompt has been parsed and the first output token has been generated, subsequent tokens are generated autoregressively (i.e., generating one token at a time), and most of the time is spent on matrix-vector multiplications.
Matrix-vector multiplications are highly _bandwidth bound_ because each neural network weight is used in only a single multiply-accumulate (MAC) operation (in contrast to LLM training and prompt parsing, which involve more matrix-matrix multiplications and are often compute-bound).

**Existing approaches** to address the memory bottleneck in LLM inference, especially on edge devices, involve:

1. weight quantization and
2. speculative execution.

The method demonstrated in this repository is related to weight quantization but takes the idea further by combining it with modern techniques for lossless data compression (aka entropy coding).
Rather than just rounding each neural network weight to a fixed precision (e.g., 4 bit per weight), as is done in weight quantization, the demos in this repository load the matrices on the GPU in compressed form (think "gzip" but faster and more effective).
We decompress the matrix entries on-the-fly directly on the GPU during the matrix-vector multiplication.
Since each decompressed matrix entry is used in only a single MAC operation, we can then immediately throw it away, and we never have to hold the entire decompressed matrix in memory.

Despite the additional computation that is necessary for decompression, preliminary results indicate that our approach ends up being faster than performing the same matrix-vector multiplications in uncompressed form.
This can be explained by the fact that entropy coding compresses the matrices to a shorter bit string, thus reducing the number of memory accesses by the GPU.
Also, modern entropy coders can decompress data extremely fast when using adequate entropy models (our demos currently use an ANS entropy coder [[2](https://ieeexplore.ieee.org/abstract/document/7170048), [3](https://arxiv.org/pdf/2201.01741)]).


## References

1. Ivanov, Andrei, et al. "[Data movement is all you need: A case study on optimizing transformers.](https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf)" Proceedings of Machine Learning and Systems 3 (2021): 711-732.
2. Duda, Jarek, et al. "[The use of asymmetric numeral systems as an accurate replacement for Huffman coding.](https://ieeexplore.ieee.org/abstract/document/7170048)" 2015 Picture Coding Symposium (PCS). IEEE, 2015.
3. Bamler, Robert. "[Understanding entropy coding with asymmetric numeral systems (ans): a statistician's perspective.](https://arxiv.org/pdf/2201.01741)" arXiv preprint arXiv:2201.01741 (2022).
