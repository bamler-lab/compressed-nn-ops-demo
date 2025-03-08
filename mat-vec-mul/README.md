# Compressed Matrix-Vector Multiplication

This directory will contain a simple demo website that runs a custom matrix multiplication kernel on the client's GPU using WebGL.
The custom kernel requires us to load only a compressed (i.e., entropy coded) form of the matrix onto the GPU, thus reducing the strain on GPU bandwidth.
The GPU kernel then uses a parallel entropy coder to decode the matrix on the fly as it is performing the matrix-vector multiplication, thus eliminating the need to ever write the decompressed matrix to memory.

## How to run

I'm not yet done implementing this demo.
