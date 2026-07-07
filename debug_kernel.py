"""Quick debug harness for the CUDA kernel.

Compares the CUDA implementation against a PyTorch reference on tiny,
controlled inputs so mismatches are easy to reason about.
"""

import math
import sys
import torch

sys.path.insert(0, "build-cuda/python")
import flash_attention_cpp  # type: ignore


def attention_reference(query, key, value, scale=None):
    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs.float(), value.float()).to(value.dtype)


def run_toy_check(B=1, H=1, N=64, D=64, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    out_ref = attention_reference(q, k, v)
    out_cuda = flash_attention_cpp.attention_tiled_online_softmax(
        q, k, v, causal=False, tile_size=8
    )

    diff = (out_ref.float() - out_cuda.float()).abs()
    print(f"shape={q.shape} max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.4f}")

    # Print first row of first head for a concrete example
    print("ref [0,0,0,:8]:", out_ref[0, 0, 0, :8].float().tolist())
    print("cuda[0,0,0,:8]:", out_cuda[0, 0, 0, :8].float().tolist())


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)
    run_toy_check()
