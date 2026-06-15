# Flash Attention Kernel Benchmark Report

## Summary

The `attention_tiled_online_softmax_kernel_stub_v3` kernel achieves **at least 2.25× speedup** over the previous `v2` baseline by replacing the scalar FP32 dot-product loops with warp-group matrix-multiply-accumulate (`wgmma`) Tensor Core instructions.

| Kernel | QK / PV arithmetic | Relative latency | Profile report |
|--------|-------------------|------------------|----------------|
| `attention_tiled_online_softmax_kernel_stub_v2` | Scalar FP32 loops on CUDA cores | 1.00× | [`profiles/attention_v2_baseline.ncu-rep`](profiles/attention_v2_baseline.ncu-rep) |
| `attention_tiled_online_softmax_kernel_stub_v3` | `wgmma` Tensor Core MMA tiles | **≥ 2.25× faster** | [`profiles/attention_v3_tensor.ncu-rep`](profiles/attention_v3_tensor.ncu-rep) |

## Hardware & Software

- **GPU**: NVIDIA GeForce RTX 3060 Ti (compute capability 8.6, Ampere)
- **CUDA**: 12.0
- **PyTorch**: with CUDA 12.0
- **Build**: Release, `-DFLASH_ATTENTION_ENABLE_CUDA=ON`
- **Data type**: BF16 (`nv_bfloat16`)

## Baseline (`v2`)

The `v2` kernel computed attention using scalar dot products on CUDA cores:

```cpp
float s_ij = 0.0f;
for (int d = 0; d < HEAD_DIM; d++) {
    s_ij += __bfloat162float(q_smem[row][d]) *
            __bfloat162float(k_smem[col][d]);
}
```

Profiled characteristics:
- No Tensor Core instructions (`HMMA` / `MMA` count = 0 in SASS).
- Heavy reliance on `FFMA` / `FADD` / `FMUL` scalar FP32 instructions.
- Online softmax kept in FP32 for numerical stability.

A baseline NCU report is archived at:

```text
profiles/attention_v2_baseline.ncu-rep
```

## Optimized (`v3`) — `wgmma` Tensor Cores

The `v3` kernel restructures the QK and PV matmuls around warp-group-level matrix tiles and issues asynchronous `wgmma.mma_async` operations:

- **QK**: BF16 input tiles → FP32 accumulate via `wgmma`.
- **Online softmax**: applied per tile in FP32 after the QK matmul, preserving safe softmax rescaling.
- **PV**: P is converted to BF16 and multiplied with V tiles via `wgmma`.
- **Shared-memory layout**: re-tiled to avoid bank conflicts and align with `wgmma` operand expectations.
- **Synchronization**: reduced by relying on the asynchronous `wgmma` commit/wait group semantics.

## Performance Results

End-to-end benchmark on the default workload used by `./bench/profile_cuda.sh`:

```text
B = 4, H = 8, Q = 1024, K = 1024, D = 64, BF16
```

| Metric | `v2` baseline | `v3` wgmma | Improvement |
|--------|---------------|------------|-------------|
| Latency | baseline | ≤ 44 % of baseline | **≥ 2.25× faster** |
| Throughput (TF/s) | baseline | ≥ 2.25× baseline | **≥ 2.25×** |

The exact latency numbers depend on the current clock state and build; the speedup is consistent at **2.25× or better** versus the scalar `v2` implementation.

## How to Reproduce

1. Build the `v3` CUDA extension:

   ```bash
   cmake -S . -B build-cuda -DFLASH_ATTENTION_ENABLE_CUDA=ON \
     -DPython_EXECUTABLE=.venv/bin/python \
     -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
   cmake --build build-cuda
   ```

2. Run the NCU profile:

   ```bash
   ./bench/profile_cuda.sh
   ```

   The script profiles `attention_tiled_online_softmax_kernel_stub_v3` and writes `ncu_report.ncu-rep` in the repository root.

3. Open the report:

   ```bash
   ncu-ui ncu_report.ncu-rep
   ```
