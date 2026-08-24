import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn.attention.flex_attention import create_mask

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.utils.local_ca import auto_local_ca_mask


def make_decoder(window_size: int, window_wrap: bool, num_queries: int) -> MaskFormerDecoder:
    decoder_layer_config = {
        "dim": 64,
        "norm": "LayerNorm",
        "dense_kwargs": {},
        "attn_kwargs": {"attn_type": "flex"},
        "bidirectional_ca": True,
        "hybrid_norm": False,
    }
    return MaskFormerDecoder(
        num_queries=num_queries,
        decoder_layer_config=decoder_layer_config,
        num_decoder_layers=1,
        mask_attention=False,
        local_strided_attn=True,
        window_size=window_size,
        window_wrap=window_wrap,
    )


def normalize_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"Expected singleton head dimension, got shape={tuple(mask.shape)}")
        mask = mask.squeeze(1)
    return mask


def build_flex_mask(
    decoder: MaskFormerDecoder,
    q_len: int,
    kv_len: int,
    stride_q_len: int,
    valid_q_len: int,
    device: str,
) -> torch.Tensor:
    block_mask = decoder.flex_local_ca_mask(
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        dtype_float=torch.float32,
        stride_q_len=stride_q_len,
        valid_q_len=valid_q_len,
    )
    return normalize_mask_shape(create_mask(block_mask.mask_mod, 1, 1, q_len, kv_len, device))


def build_expected_mask_lt_q_len(q_len: int, kv_len: int, num_selected: int, window_size: int, wrap: bool) -> torch.Tensor:
    expected_selected = auto_local_ca_mask(
        q=torch.empty(1, num_selected),
        kv=torch.empty(1, kv_len),
        window_size=window_size,
        wrap=wrap,
    )
    expected = torch.zeros((1, q_len, kv_len), dtype=torch.bool)
    expected[:, :num_selected, :] = expected_selected
    return expected


def build_expected_mask_eq_q_len(q_len: int, kv_len: int, window_size: int, wrap: bool) -> torch.Tensor:
    return auto_local_ca_mask(
        q=torch.empty(1, q_len),
        kv=torch.empty(1, kv_len),
        window_size=window_size,
        wrap=wrap,
    )


def plot_masks_for_wrap(
    output_dir: Path,
    q_len: int,
    kv_len: int,
    num_selected: int,
    window_size: int,
    wrap: bool,
    device: str,
) -> Path:
    decoder = make_decoder(window_size=window_size, window_wrap=wrap, num_queries=q_len)

    flex_lt = build_flex_mask(
        decoder=decoder,
        q_len=q_len,
        kv_len=kv_len,
        stride_q_len=num_selected,
        valid_q_len=num_selected,
        device=device,
    )
    expected_lt = build_expected_mask_lt_q_len(
        q_len=q_len,
        kv_len=kv_len,
        num_selected=num_selected,
        window_size=window_size,
        wrap=wrap,
    )

    flex_eq = build_flex_mask(
        decoder=decoder,
        q_len=q_len,
        kv_len=kv_len,
        stride_q_len=q_len,
        valid_q_len=q_len,
        device=device,
    )
    expected_eq = build_expected_mask_eq_q_len(
        q_len=q_len,
        kv_len=kv_len,
        window_size=window_size,
        wrap=wrap,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], flex_lt[0], f"Flex: num_selected={num_selected} < q_len={q_len}"),
        (axes[0, 1], expected_lt[0], "Expected (selected rows + padded zeros)"),
        (axes[1, 0], flex_eq[0], f"Flex: num_selected=q_len={q_len}"),
        (axes[1, 1], expected_eq[0], "Expected (full local mask)"),
    ]

    for ax, mask, title in panels:
        ax.imshow(mask.to(torch.int32).cpu().numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")

    fig.suptitle(f"Flex Local CA Masks (wrap={wrap})", fontsize=13)

    output_path = output_dir / f"flex_local_ca_dynamic_wrap_{int(wrap)}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot flex local cross-attention masks for dynamic query settings.")
    parser.add_argument("--q-len", type=int, default=12, help="Total query length.")
    parser.add_argument("--kv-len", type=int, default=60, help="Key/value length.")
    parser.add_argument("--num-selected", type=int, default=6, help="Number of selected (unpadded) queries.")
    parser.add_argument("--window-size", type=int, default=20, help="Local attention window size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device passed to mask creation.")
    parser.add_argument("--output-dir", type=Path, default=Path("tests/outputs/flex"), help="Directory for output PNGs.")
    parser.add_argument(
        "--wrap-mode",
        choices=["both", "false", "true"],
        default="both",
        help="Whether to plot for wrapped windows, non-wrapped windows, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_selected < 1:
        raise ValueError(f"--num-selected must be >= 1, got {args.num_selected}")
    if args.num_selected > args.q_len:
        raise ValueError(f"--num-selected ({args.num_selected}) cannot exceed --q-len ({args.q_len})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    wrap_values = [False, True] if args.wrap_mode == "both" else [args.wrap_mode == "true"]
    for wrap in wrap_values:
        out_path = plot_masks_for_wrap(
            output_dir=args.output_dir,
            q_len=args.q_len,
            kv_len=args.kv_len,
            num_selected=args.num_selected,
            window_size=args.window_size,
            wrap=wrap,
            device=args.device,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
