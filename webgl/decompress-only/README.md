# Entropy Decoding in WebGL

This directory contains a simple demo website that runs a massively parallel entropy coder on the client's GPU using WebGL.
The demo tests an extreme situation: 1 million entropy coders that decode in parallel, all reading from the same bit string, and with a synchronized interruption for (quite computationally expensive) bookkeeping after every time we've decoded a single symbol on each thread.
Further, all decoded data is written to (GPU) memory, which would not be necessary in many machine-learning applications (as, e.g., decoded weights could instead be immediately used and then dropped).

The decoded data is verified for correctness using fairly strong checksums.

## How to run

To test this demo, open a terminal, `cd` into this directory, and then execute

```bash
python -m http.server
```

Then, point your web browser to the URL that is printed to the terminal (usually <http://0.0.0.0:8000/>).

## Performance

When decoding random data with an entropy of about 5 bit per symbol, the test reaches a throughput of about 1 billion decoded symbols per second on my model 2021 laptop with integrated intel GPU on Firefox.
The recorded runtime includes the time for uploading compressed data to the GPU, but not for downloading decompressed data from the GPU (we do read one symbol from the last batch of decoded symbols though before stopping the timer to ensure that decoding has really finished and isn't continuing asynchronously on the GPU).

A lot of the computing time is currently spent on bookkeeping (for demuxing the compressed bit string across the 1 million parallel entropy coders), and probably on memory writes.
Both of these overheads can probably be reduced significantly if we decode many symbols per thread between bookkeeping interruptions, and if we immediately use the decoded data for calculations rather than writing it out to memory.
