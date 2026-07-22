"""Run attention benchmarks and generate comparison plots + a markdown report.

Measures, per implementation and per sequence length:
- median latency (seconds)
- throughput (TFLOP/s)
- peak extra memory used by a single forward pass (MiB)

Outputs (relative to the repo root):
- bench/results/benchmark_results.csv
- bench/plots/{latency,tflops,memory}.png
- bench/BENCHMARK_RESULTS.md

Usage (from repo root):
    PYTHONPATH=build/python:$PYTHONPATH python bench/plot_benchmark.py
    # optional:
    python bench/plot_benchmark.py --device cpu --dtype fp32 --seq-lens 128 256 512
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import resource
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.bench_attention import _attention_flops, _bench_one  # noqa: E402
from src.flash_attention import (  # noqa: E402
    attention_reference,
    attention_tiled_online_softmax_cpp,
    attention_tiled_online_softmax_reference,
)

RESULTS_DIR = REPO_ROOT / "bench" / "results"
PLOTS_DIR = REPO_ROOT / "bench" / "plots"
REPORT_PATH = REPO_ROOT / "bench" / "BENCHMARK_RESULTS.md"


def _measure_peak_memory_mb(fn, query, key, value, kwargs) -> float:
    """Peak extra memory of one forward pass, in MiB (best-effort)."""
    if query.device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(query.device)
        torch.cuda.reset_peak_memory_stats(query.device)
        fn(query, key, value, **kwargs)
        torch.cuda.synchronize(query.device)
        peak = torch.cuda.max_memory_allocated(query.device)
        return (peak - base) / 1e6
    # CPU fallback: delta of process peak RSS (Linux ru_maxrss is in KiB).
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fn(query, key, value, **kwargs)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max(0.0, (after - before) / 1024.0)


def _sdpa(q, k, v, *, causal=False):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _build_impls() -> list[tuple[str, object, dict]]:
    return [
        ("pytorch_reference", attention_reference, {}),
        (
            "tiled_python",
            attention_tiled_online_softmax_reference,
            {"tile_size": True},
        ),
        ("torch_sdpa", _sdpa, {}),
        ("cpp_extension", attention_tiled_online_softmax_cpp, {"tile_size": True}),
    ]


def run_benchmarks(
    *,
    seq_lens: list[int],
    batch: int,
    heads: int,
    dim: int,
    causal: bool,
    tile_size: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    rows: list[dict] = []
    generator = torch.Generator().manual_seed(42)

    for seq_len in seq_lens:
        query = torch.randn(
            batch, heads, seq_len, dim, dtype=dtype, generator=generator
        ).to(device)
        key = torch.randn(
            batch, heads, seq_len, dim, dtype=dtype, generator=generator
        ).to(device)
        value = torch.randn(
            batch, heads, seq_len, dim, dtype=dtype, generator=generator
        ).to(device)

        for name, fn, opts in _build_impls():
            kwargs = {"causal": causal}
            if opts.get("tile_size"):
                kwargs["tile_size"] = tile_size
            try:
                bench = _bench_one(
                    fn,
                    query,
                    key,
                    value,
                    causal=causal,
                    tile_size=tile_size,
                    warmup=warmup,
                    repeats=repeats,
                    name=name,
                )
                memory_mb = _measure_peak_memory_mb(fn, query, key, value, kwargs)
            except Exception as exc:
                warnings.warn(f"{name} @ seq_len={seq_len} skipped: {exc}")
                continue
            flops = _attention_flops(batch, heads, seq_len, seq_len, dim, dim)
            rows.append(
                {
                    "seq_len": seq_len,
                    "impl": name,
                    "median_s": bench["median_s"],
                    "tflops": flops / bench["median_s"] / 1e12,
                    "memory_mb": memory_mb,
                }
            )
            print(
                f"seq_len={seq_len:>5}  {name:<20} "
                f"{bench['median_s'] * 1e3:>9.3f} ms  "
                f"{rows[-1]['tflops']:>8.3f} TF/s  {memory_mb:>9.1f} MiB"
            )
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seq_len", "impl", "median_s", "tflops", "memory_mb"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: list[dict], metric: str, ylabel: str, title: str, path: Path) -> None:
    impls = sorted({r["impl"] for r in rows})
    seq_lens = sorted({r["seq_len"] for r in rows})
    all_pow2 = all(s & (s - 1) == 0 for s in seq_lens)
    fig, ax = plt.subplots(figsize=(8, 5))
    for impl in impls:
        pts = sorted((r["seq_len"], r[metric]) for r in rows if r["impl"] == impl)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=impl)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if all_pow2:
        ax.set_xscale("log", base=2)
    else:
        ax.set_xticks(seq_lens)
        ax.tick_params(axis="x", rotation=30)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plots(rows: list[dict]) -> dict[str, Path]:
    plots = {
        "latency": PLOTS_DIR / "latency.png",
        "tflops": PLOTS_DIR / "tflops.png",
        "memory": PLOTS_DIR / "memory.png",
    }
    _plot_metric(
        rows, "median_s", "Median latency (s)", "Attention latency vs sequence length",
        plots["latency"],
    )
    _plot_metric(
        rows, "tflops", "Throughput (TFLOP/s)",
        "Attention throughput vs sequence length", plots["tflops"],
    )
    _plot_metric(
        rows, "memory_mb", "Peak extra memory (MiB)",
        "Attention memory usage vs sequence length", plots["memory"],
    )
    return plots


def _markdown_table(rows: list[dict], metric: str, fmt) -> str:
    impls = sorted({r["impl"] for r in rows})
    seq_lens = sorted({r["seq_len"] for r in rows})
    lookup = {(r["seq_len"], r["impl"]): r[metric] for r in rows}
    header = "| seq_len | " + " | ".join(impls) + " |"
    sep = "|" + "---|" * (len(impls) + 1)
    lines = [header, sep]
    for s in seq_lens:
        cells = [fmt(lookup[(s, i)]) if (s, i) in lookup else "-" for i in impls]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_report(rows: list[dict], plots: dict[str, Path], context: str) -> None:
    def ms(v):
        return f"{v * 1e3:.3f}"

    def tf(v):
        return f"{v:.3f}"

    def mib(v):
        return f"{v:.1f}"

    report = f"""# Attention Benchmark Results

{context}

## Latency (median, ms)

![Latency](plots/latency.png)

{_markdown_table(rows, "median_s", ms)}

## Throughput (TFLOP/s)

![Throughput](plots/tflops.png)

{_markdown_table(rows, "tflops", tf)}

## Peak extra memory (MiB)

![Memory usage](plots/memory.png)

{_markdown_table(rows, "memory_mb", mib)}

---

Generated by `bench/plot_benchmark.py`. Raw data: `bench/results/benchmark_results.csv`.
"""
    REPORT_PATH.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark attention kernels and plot the results"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=("fp16", "fp32", "bf16"))
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = (
        dtype_map[args.dtype]
        if args.dtype
        else (torch.bfloat16 if device.type == "cuda" else torch.float32)
    )

    context = (
        f"Config: B={args.batch}, H={args.heads}, D={args.dim}, "
        f"causal={args.causal}, tile_size={args.tile_size}, "
        f"device={device}, dtype={dtype}, warmup={args.warmup}, repeats={args.repeats}."
    )
    print(context)

    rows = run_benchmarks(
        seq_lens=args.seq_lens,
        batch=args.batch,
        heads=args.heads,
        dim=args.dim,
        causal=args.causal,
        tile_size=args.tile_size,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
        dtype=dtype,
    )
    if not rows:
        sys.exit("No benchmark results collected; nothing to plot.")

    save_csv(rows, RESULTS_DIR / "benchmark_results.csv")
    plots = save_plots(rows)
    save_report(rows, plots, context)

    print(f"\nCSV:    {RESULTS_DIR / 'benchmark_results.csv'}")
    print(f"Plots:  {PLOTS_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
