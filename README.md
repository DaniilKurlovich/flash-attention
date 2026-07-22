# Flash Attention

## Scope

This repository implement combine version of Flash Attention papers (v1, v2)

Implemented today:

- `attention_reference`: direct scaled dot-product attention over `B x H x T x D` tensors
- `attention_tiled_online_softmax_reference`: a tiled reference using the online softmax recurrence
- `attention_tiled_online_softmax_cpp`: a compiled C++ wrapper with the same Python call signature as the tiled reference
- `attention_tiled_online_softmax_cuda`: a fused CUDA kernel with tiled Q/K/V loads, online softmax, and causal masking
- a native C++ test binary that checks the implementations agree under causal and non-causal settings

Not in scope yet:

- custom autograd
- dropout, bias terms, packed sequence handling, or kernel auto-tuning

## Invariants

- Inputs use shape `(batch, heads, seqlen, dim)` for `query` and `key`
- `value` uses shape `(batch, heads, seqlen, value_dim)`
- `query` and `key` share the same head dimension
- the tiled implementation must match the direct reference numerically within standard floating-point tolerance
- accumulation is done in `float32` for stability, then cast back to the value dtype
- causal mode means position `i` may only attend to keys `<= i`

## Running Tests

To build and run the native C++ test binary:

```bash
cmake -S . -B build
cmake --build build --target flash_attention_csrc_tests
./build/flash_attention_csrc_tests
```

To build the C++ extension with CMake:

```bash
cmake -S . -B build
cmake --build build
```

To build with CUDA support (uses the CUDA kernel when tensors are on GPU):

```bash
source .venv/bin/activate
cmake -S . -B build-cuda -DFLASH_ATTENTION_ENABLE_CUDA=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda
```

The CUDA kernel currently requires `head_dim == 64` and `dtype == bfloat16`.

## Benchmarking

A benchmark harness lives in `bench/bench_attention.py` and compares:

- PyTorch reference (`attention_reference`)
- tiled Python reference (`attention_tiled_online_softmax_reference`)
- C++ / CUDA extension (`attention_tiled_online_softmax_cpp`)
- `torch.nn.functional.scaled_dot_product_attention`

Run a single configuration:

```bash
source .venv/bin/activate
PYTHONPATH=build-cuda/python:$PYTHONPATH python3 bench/bench_attention.py \
  --batch 4 --heads 8 --q-len 3638 --dim 64 --repeats 30
```

Sweep over sequence lengths:

```bash
PYTHONPATH=build-cuda/python:$PYTHONPATH python3 bench/bench_attention.py --sweep
```

## Profiling

### Nsight Compute profiler

`bench/profile_cuda.sh` runs an Nsight Compute (`ncu`) session against the CUDA attention kernel and writes `ncu_report.ncu-rep` in the repository root:

```bash
./bench/profile_cuda.sh
```

## Project Layout

```
src/              Python modules and C++ / CUDA sources
src/csrc/         C++ and CUDA kernel implementations
src/csrc/tests/   native C++ correctness tests
bench/            benchmark and profiling scripts
notes/            design notes and research writeups
build/            default CMake build directory (CPU / CUDA)
build-cuda/       CMake build directory for CUDA-enabled extension
```
