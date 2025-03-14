# GPU Kernels That Decompress Matrices on the Fly

This repository contains some proof-of-concept implementations of entropy coding on the GPU.
The eventual goal is to speed up memory-bound linear algebra operations by moving matrices to the GPU in compressed (i.e., entropy coded) representation and decoding it on the fly while operating on it.

See subdirectories for examples.
The implementation in this repository are (slow) prototypes meant for experimentation and for settling on a data format for compressed matrices.
Faster implementation in CUDA and in WebGPU comput shaders are WIP.
