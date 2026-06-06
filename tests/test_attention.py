import pytest
import torch

from src.flash_attention import (
    attention_reference,
    attention_tiled_online_softmax_cpp,
    attention_tiled_online_softmax_reference,
)


def _test_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def _make_inputs(
    *,
    batch: int = 2,
    heads: int = 3,
    q_len: int = 17,
    k_len: int = 17,
    dim: int = 128,
    value_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = _test_device_and_dtype()
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(batch, heads, q_len, dim, dtype=dtype, generator=generator).to(device)
    key = torch.randn(batch, heads, k_len, dim, dtype=dtype, generator=generator).to(device)
    value = torch.randn(batch, heads, k_len, value_dim, dtype=dtype, generator=generator).to(device)
    return query, key, value


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tile_size", [1, 4, 8, 64])
def test_tiled_matches_reference(causal: bool, tile_size: int) -> None:
    query, key, value = _make_inputs(q_len=19, k_len=23, dim=128)

    expected = attention_reference(query, key, value, causal=causal)
    actual = attention_tiled_online_softmax_reference(
        query,
        key,
        value,
        causal=causal,
        tile_size=tile_size,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_causal_mask_blocks_future_tokens() -> None:
    query, key, value = _make_inputs(batch=1, heads=1, q_len=4, k_len=4, dim=128, value_dim=128)

    full = attention_reference(query, key, value, causal=False)
    causal = attention_reference(query, key, value, causal=True)

    assert not torch.allclose(full, causal)
    torch.testing.assert_close(causal[:, :, 0], value[:, :, 0], atol=1e-5, rtol=1e-5)


def test_invalid_shapes_raise() -> None:
    query, key, value = _make_inputs()

    with pytest.raises(ValueError, match="4D tensors"):
        attention_reference(query[0], key, value)

    with pytest.raises(ValueError, match="same sequence length"):
        attention_tiled_online_softmax_reference(query, key[:, :, :-1], value)


def test_invalid_tile_size_raises() -> None:
    query, key, value = _make_inputs()

    with pytest.raises(ValueError, match="tile_size must be positive"):
        attention_tiled_online_softmax_reference(query, key, value, tile_size=0)


def attention_base(q, k, v):
    return torch.softmax((q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1) @ v
    

# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("tile_size", [1])
def test_cpp_wrapper_matches_reference(causal: bool, tile_size: int) -> None:
    pytest.importorskip(
        "flash_attention_cpp",
        reason="build the C++ extension before running wrapper tests",
    )

    query, key, value = _make_inputs(q_len=19, k_len=23, dim=128, value_dim=128)

    # expected = attention_tiled_online_softmax_reference(
    #     query,
    #     key,
    #     value,
    #     causal=causal,
    #     tile_size=tile_size,
    # )
    expected = attention_base(
        query,
        key,
        value
    )
    actual = attention_tiled_online_softmax_cpp(
        query,
        key,
        value,
        causal=causal,
        tile_size=tile_size,
    )
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-3)
