# Compressed Matrix-Vector Multiplication

This directory will contain a simple demo website that runs a custom matrix multiplication kernel on the client's GPU using WebGL.
The custom kernel requires us to load only a compressed (i.e., entropy coded) form of the matrix onto the GPU, thus reducing the strain on GPU bandwidth.
The GPU kernel then uses a parallel entropy coder to decode the matrix on the fly as it is performing the matrix-vector multiplication, thus eliminating the need to ever write the decompressed matrix to memory.

## How to run

1. Generate some mock compressed quantized matrices by running all cells in the notebook [mock-data.ipynb](./mock-data.ipynb).
2. Open a terminal, `cd` into this directory, and execute

   ```bash
   python -m http.server
   ```

   Then, point your web browser to the URL that is printed to the terminal (usually <http://0.0.0.0:8000/>).

## Disclaimer

This is just an initial proof of concept, faster implementations are to follow.
The current implementation is very slow (about 1G MAC per second in a heavy bandwidth-bound computation) due to:

- limitations of WebGL (which does not expose the memory hierarchy of GPUs and therefore requires a _lot_ of redundant copies); and
- small matrix sizes in this toy demo, which limit parallelism (if you're really curious, you can increase the matrix sizes in [mock-data.ipynb](./mock-data.ipynb), but for a serious performance analysis, this toy implementation is not very interesting).

We're working on implementations in CUDA (for NVIDIA GPUs) and in WebGPU compute shaders (for GPUs on most end-user devices), and we expect them to be much faster.
