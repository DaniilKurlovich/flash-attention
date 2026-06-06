"""Benchmark Flash Attention implementations.

Usage (from repo root):
    PYTHONPATH=build/python:$PYTHONPATH python bench/bench_attention.py

Comparisons:
- PyTorch reference (pure Python)
- Tiled Python reference (online softmax)
- C++ extension (CPU)
- CUDA kernel (if built with -DFLASH_ATTENTION_ENABLE_CUDA=ON)
"""

from __future__ import annotations

import math
import time
import warnings

import torch

from src.flash_attention import (
    attention_reference,
    attention_tiled_online_softmax_cpp,
    attention_tiled_online_softmax_reference,
)


def _fmt_us(t_s: float) -> str:
    if t_s < 1e-6:
        return f"{t_s * 1e9:>7.2f} ns"
    if t_s < 1e-3:
        return f"{t_s * 1e6:>7.2f} us"
    return f"{t_s * 1e3:>7.2f} ms"


def _attention_flops(batch: int, heads: int, q_len: int, k_len: int, dim: int, value_dim: int) -> float:
    """Return FLOPs for a single attention forward pass."""
    # Q @ K^T
    flops_qk = 2.0 * batch * heads * q_len * k_len * dim
    # softmax + masking are bandwidth-bound, usually ignored in FLOP counts
    # P @ V
    flops_pv = 2.0 * batch * heads * q_len * k_len * value_dim
    return flops_qk + flops_pv


def _bench_one(
    fn,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    tile_size: int = 64,
    warmup: int = 3,
    repeats: int = 10,
    name: str = "unknown",
) -> dict:
    """Run a function and return median latency + throughput."""
    device = query.device
    is_cuda = device.type == "cuda"

    kwargs = {"causal": causal}
    if "tile_size" in fn.__code__.co_varnames:
        kwargs["tile_size"] = tile_size

    # Warmup
    for _ in range(warmup):
        out = fn(query, key, value, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(device)

    # Timed runs
    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            start.record()
            out = fn(query, key, value, **kwargs)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end) / 1000.0)  # ms -> s
    else:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn(query, key, value, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times.sort()
    median_s = times[len(times) // 2]
    flops = _attention_flops(
        query.shape[0], query.shape[1], query.shape[2], key.shape[2], query.shape[3], value.shape[3]
    )
    tflops = (flops / median_s) / 1e12
    return {
        "name": name,
        "median_s": median_s,
        "tflops": tflops,
        "times": times,
        "shape": tuple(out.shape),
    }


def benchmark(
    *,
    batch: int = 2,
    heads: int = 8,
    q_len: int = 1024,
    k_len: int | None = None,
    dim: int = 64,
    value_dim: int | None = None,
    causal: bool = False,
    tile_size: int = 64,
    warmup: int = 3,
    repeats: int = 10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    k_len = k_len or q_len
    value_dim = value_dim or dim

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"{'=' * 70}")
    print(f"Config: B={batch}, H={heads}, Q={q_len}, K={k_len}, D={dim}, V={value_dim}")
    print(f"        causal={causal}, tile_size={tile_size}, device={device}, dtype={dtype}")
    print(f"{'=' * 70}")

    generator = torch.Generator().manual_seed(42)
    query = torch.randn(batch, heads, q_len, dim, dtype=dtype, generator=generator).to(device)
    key = torch.randn(batch, heads, k_len, dim, dtype=dtype, generator=generator).to(device)
    value = torch.randn(batch, heads, k_len, value_dim, dtype=dtype, generator=generator).to(device)

    results = []

    # 1. PyTorch reference
    results.append(
        _bench_one(
            attention_reference,
            query, key, value,
            causal=causal,
            name="pytorch_reference",
            warmup=warmup,
            repeats=repeats,
        )
    )

    # 2. Tiled Python reference
    results.append(
        _bench_one(
            attention_tiled_online_softmax_reference,
            query, key, value,
            causal=causal,
            tile_size=tile_size,
            name="tiled_python",
            warmup=warmup,
            repeats=repeats,
        )
    )

    # 3. C++ / CUDA extension
    try:
        results.append(
            _bench_one(
                attention_tiled_online_softmax_cpp,
                query, key, value,
                causal=causal,
                tile_size=tile_size,
                name="cpp_extension",
                warmup=warmup,
                repeats=repeats,
            )
        )
    except RuntimeError as exc:
        warnings.warn(f"C++ extension not available: {exc}")

    # 4. torch.nn.functional.scaled_dot_product_attention (if available)
    try:
        def _sdpa(q, k, v, *, causal=False):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal
            )
        results.append(
            _bench_one(
                _sdpa,
                query, key, value,
                causal=causal,
                name="torch_sdpa",
                warmup=warmup,
                repeats=repeats,
            )
        )
    except Exception as exc:
        warnings.warn(f"torch sdpa skipped: {exc}")

    print(f"{'Implementation':<20} {'Median':>12} {'Throughput':>12} {'Output shape'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<20} {_fmt_us(r['median_s']):>12} {r['tflops']:>10.2f} TF/s  {r['shape']}"
        )
    print()


def sweep_sequence_lengths(
    seq_lengths: list[int] | None = None,
    **common_kwargs,
) -> None:
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]
    for seq_len in seq_lengths:
        benchmark(q_len=seq_len, k_len=seq_len, **common_kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark attention kernels")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=1024)
    parser.add_argument("--k-len", type=int, default=None)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=("fp16", "fp32", "bf16"))
    parser.add_argument("--sweep", action="store_true", help="Sweep over sequence lengths")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype] if args.dtype else None

    if args.sweep:
        sweep_sequence_lengths(
            batch=args.batch,
            heads=args.heads,
            dim=args.dim,
            value_dim=args.value_dim,
            causal=args.causal,
            tile_size=args.tile_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            dtype=dtype,
        )
    else:
        benchmark(
            batch=args.batch,
            heads=args.heads,
            q_len=args.q_len,
            k_len=args.k_len,
            dim=args.dim,
            value_dim=args.value_dim,
            causal=args.causal,
            tile_size=args.tile_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            dtype=dtype,
        )
