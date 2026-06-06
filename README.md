# Flash Attention

## Scope

This repository starts with a day-1 correctness baseline for Flash Attention in pure PyTorch and adds a fused CUDA kernel path on top of it. The goal is to keep a readable reference implementation while experimenting with a high-performance kernel.

Implemented today:

- `attention_reference`: direct scaled dot-product attention over `B x H x T x D` tensors
- `attention_tiled_online_softmax_reference`: a tiled reference using the online softmax recurrence
- `attention_tiled_online_softmax_cpp`: a compiled C++ wrapper with the same Python call signature as the tiled reference
- `attention_tiled_online_softmax_cuda`: a fused CUDA kernel with tiled Q/K/V loads, online softmax, and causal masking
- numerical `pytest` coverage that checks both paths agree under causal and non-causal settings

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

With the local virtual environment active:

```bash
python -m pytest tests
```

To build the C++ extension with CMake:

```bash
cmake -S . -B build
cmake --build build
PYTHONPATH=build/python python -m pytest tests
```

To build with CUDA support (uses the CUDA kernel when tensors are on GPU):

```bash
source .venv/bin/activate
cmake -S . -B build-cuda -DFLASH_ATTENTION_ENABLE_CUDA=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-cuda
PYTHONPATH=build-cuda/python python -m pytest tests
```

The CUDA kernel currently requires `head_dim == 128` and `dtype == float16`.

To build and run the native C++ test binary:

```bash
cmake -S . -B build
cmake --build build --target flash_attention_csrc_tests
./build/flash_attention_csrc_tests
```

## Benchmarking

A `pytest`-style benchmark harness lives in `bench/bench_attention.py` and compares:

- PyTorch reference (`attention_reference`)
- tiled Python reference (`attention_tiled_online_softmax_reference`)
- C++ / CUDA extension (`attention_tiled_online_softmax_cpp`)
- `torch.nn.functional.scaled_dot_product_attention`

Run a single configuration:

```bash
source .venv/bin/activate
PYTHONPATH=build-cuda/python:$PYTHONPATH python bench/bench_attention.py \
  --batch 2 --heads 8 --q-len 1024 --dim 128 --repeats 10
```

Sweep over sequence lengths:

```bash
PYTHONPATH=build-cuda/python:$PYTHONPATH python bench/bench_attention.py --sweep
```

## Profiling

### Unified CUDA + memory profiler

`bench/profile_all.sh` runs a single Nsight Compute (`ncu`) session and prints both occupancy/register metrics and a full memory-bandwidth hierarchy diagram:

```bash
source .venv/bin/activate
./bench/profile_all.sh
```

Example output:

```
╔══════════════════════════════════════════════════════════════════════╗
║                     CUDA Kernel Profile Summary                      ║
╚══════════════════════════════════════════════════════════════════════╝
  Duration      : 0.456 ms
  Occupancy     : 62.5%
  Registers     : 64 per thread
  Static smem   : 17408 bytes
  Global LD eff : 87.5%

╔══════════════════════════════════════════════════════════════════════╗
║           GPU Memory Bandwidth Hierarchy (per kernel launch)         ║
╚══════════════════════════════════════════════════════════════════════╝
  ...
```

The script writes `ncu_report.ncu-rep` for inspection in `ncu-ui`:

```bash
ncu-ui ncu_report.ncu-rep
```

### Source-level hotspot analysis

The CUDA build embeds `-lineinfo`, so the Nsight Compute **Source** page maps SASS instructions back to `src/csrc/attention_tiled_cuda.cu`. Use it to identify hotspots in:

- QK^T compute (around line 115)
- causal / bounds mask logic (around line 127)
- online softmax update (around line 134)
- PV accumulation (around line 177)

### Permission fix for `ncu`

If you get `ERR_NVGPUCTRPERM`:

```bash
sudo chmod 666 /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2
```

## Project Layout

```
src/              Python modules and C++ / CUDA sources
src/csrc/         C++ and CUDA kernel implementations
tests/            pytest correctness tests
bench/            benchmark and profiling scripts
notes/            design notes and research writeups
build/            default CMake build directory (CPU / CUDA)
build-cuda/       CMake build directory for CUDA-enabled extension
```
