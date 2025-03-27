# WebGL Implementations

**This directory contains experimental code that pushes the limits of portability (at the expense of performance).
The more relevant demos are in the directory [`wgpu`](../wgpu).**

This directory contains implementations of GPU shaders in WebGL that run directly inside a browser.
The directory [`decompress-only`](./decompress-only) demonstrates decompression of an entire sequence of matrices on the GPU, while the directory [`mat-vec-mul`](./mat-vec-mul) demonstrates decompression that happens on-the-fly during matrix-vector-multiplication.

The implementations in this directory "misuse" WebGL fragment shaders for general computational tasks.
This makes the code in this directory extremely portable, but it sacrifices a lot of performance since fragment shaders come with lots of restrictions (they are actually meant for use in a graphics pipeline and not for general purpose GPU computing).
For implementations in the more adequate (and still very portable) WebGPU, see the directory [`wgpu`](../wgpu).
