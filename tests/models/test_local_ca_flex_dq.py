import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.local_ca import auto_local_ca_mask


def _make_decoder(window_size: int, window_wrap: bool) -> MaskFormerDecoder:
    decoder_layer_config = {
        "dim": 64,
        "norm": "LayerNorm",
        "dense_kwargs": {},
        "attn_kwargs": {"attn_type": "flex"},
        "bidirectional_ca": True,
        "hybrid_norm": False,
    }
    return MaskFormerDecoder(
        num_queries=12,
        decoder_layer_config=decoder_layer_config,
        num_decoder_layers=1,
        mask_attention=False,
        local_strided_attn=True,
        window_size=window_size,
        window_wrap=window_wrap,
    )


def _normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    # create_mask may return (B, 1, Q, KV) or (B, Q, KV)
    if mask.dim() == 4:
        assert mask.shape[1] == 1
        mask = mask.squeeze(1)
    return mask


@pytest.mark.parametrize("window_wrap", [False, True])
def test_flex_local_ca_mask_dynamic_num_selected_lt_q_len(window_wrap: bool) -> None:
    q_len = 12
    kv_len = 60
    num_selected = 6
    window_size = 20
    device = "cpu"

    decoder = _make_decoder(window_size=window_size, window_wrap=window_wrap)
    block_mask = decoder.flex_local_ca_mask(
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype_float=torch.float32,
        stride_q_len=num_selected,
        valid_q_len=num_selected,
    )
    mask = _normalize_mask_shape(create_mask(block_mask.mask_mod, 1, 1, q_len, kv_len, device))

    expected_selected = auto_local_ca_mask(
        q=torch.empty(1, num_selected),
        kv=torch.empty(1, kv_len),
        window_size=window_size,
        wrap=window_wrap,
    )

    assert mask.shape == (1, q_len, kv_len)
    assert torch.equal(mask[:, :num_selected], expected_selected)
    assert not mask[:, num_selected:, :].any()


@pytest.mark.parametrize("window_wrap", [False, True])
def test_flex_local_ca_mask_dynamic_num_selected_eq_q_len(window_wrap: bool) -> None:
    q_len = 12
    kv_len = 60
    window_size = 20
    device = "cpu"

    decoder = _make_decoder(window_size=window_size, window_wrap=window_wrap)
    block_mask = decoder.flex_local_ca_mask(
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype_float=torch.float32,
        stride_q_len=q_len,
        valid_q_len=q_len,
    )
    mask = _normalize_mask_shape(create_mask(block_mask.mask_mod, 1, 1, q_len, kv_len, device))

    expected_full = auto_local_ca_mask(
        q=torch.empty(1, q_len),
        kv=torch.empty(1, kv_len),
        window_size=window_size,
        wrap=window_wrap,
    )

    assert mask.shape == (1, q_len, kv_len)
    assert torch.equal(mask, expected_full)
