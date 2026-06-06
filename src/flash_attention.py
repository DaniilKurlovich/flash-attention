"""Day-1 reference implementations for Flash Attention.

This module intentionally stays in pure PyTorch. It defines:

- a straightforward scaled dot-product attention reference
- a tiled reference that uses the online softmax recurrence

Both implementations operate on tensors with shape ``(batch, heads, seqlen, dim)``.
"""

from __future__ import annotations

import importlib
import math
from functools import lru_cache

import torch


def _validate_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must all be 4D tensors shaped (B, H, T, D)")

    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("query, key, and value must agree on batch and head dimensions")

    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")

    if key.shape[-1] != query.shape[-1]:
        raise ValueError("query and key must have the same head dimension")

    if value.shape[-1] <= 0:
        raise ValueError("value head dimension must be positive")


def _causal_mask(
    q_len: int,
    k_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(q_len, device=device).unsqueeze(-1)
    k_positions = torch.arange(k_len, device=device).unsqueeze(0)
    return k_positions <= q_positions


def attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention in a direct, readable form."""
    _validate_inputs(query, key, value)

    q_len = query.shape[-2]
    k_len = key.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if causal:
        mask = _causal_mask(q_len, k_len, device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
    return torch.matmul(probs.float(), value.float()).to(dtype=value.dtype)


def attention_tiled_online_softmax_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    tile_size: int = 64,
) -> torch.Tensor:
    """Compute attention with tiled key/value blocks and an online softmax update."""
    _validate_inputs(query, key, value)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    batch, heads, q_len, _ = query.shape
    k_len = key.shape[-2]
    value_dim = value.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])

    q = query.float()
    k = key.float()
    v = value.float()

    output = torch.empty((batch, heads, q_len, value_dim), device=query.device, dtype=torch.float32)
    row_max = torch.full((batch, heads, q_len), float("-inf"), device=query.device, dtype=torch.float32)
    row_sum = torch.zeros((batch, heads, q_len), device=query.device, dtype=torch.float32)
    row_acc = torch.zeros((batch, heads, q_len, value_dim), device=query.device, dtype=torch.float32)

    q_positions = torch.arange(q_len, device=query.device) if causal else None

    for start in range(0, k_len, tile_size):
        end = min(start + tile_size, k_len)
        k_tile = k[:, :, start:end, :]
        v_tile = v[:, :, start:end, :]

        scores = torch.matmul(q, k_tile.transpose(-2, -1)) * scale
        if causal:
            k_positions = torch.arange(start, end, device=query.device)
            tile_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(-1)
            scores = scores.masked_fill(~tile_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        tile_row_max = scores.amax(dim=-1)
        new_row_max = torch.maximum(row_max, tile_row_max)

        exp_row_scale = torch.exp(row_max - new_row_max)
        exp_scores = torch.exp(scores - new_row_max.unsqueeze(-1))

        row_sum = row_sum * exp_row_scale + exp_scores.sum(dim=-1)
        row_acc = row_acc * exp_row_scale.unsqueeze(-1) + torch.matmul(exp_scores, v_tile)
        row_max = new_row_max

    output.copy_(row_acc / row_sum.unsqueeze(-1))
    return output.to(dtype=value.dtype)


@lru_cache(maxsize=1)
def _load_cpp_extension():
    try:
        return importlib.import_module("flash_attention_cpp")
    except ImportError as exc:
        raise RuntimeError(
            "flash_attention_cpp is not built. Configure and build the extension with "
            "`cmake -S . -B build && cmake --build build`, then add `build/python` to PYTHONPATH."
        ) from exc


def attention_tiled_online_softmax_cpp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    tile_size: int = 64,
) -> torch.Tensor:
    """Dispatch tiled attention through the compiled C++ extension."""
    _validate_inputs(query, key, value)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    extension = _load_cpp_extension()
    return extension.attention_tiled_online_softmax(
        query,
        key,
        value,
        causal=causal,
        scale=scale,
        tile_size=tile_size,
    )
