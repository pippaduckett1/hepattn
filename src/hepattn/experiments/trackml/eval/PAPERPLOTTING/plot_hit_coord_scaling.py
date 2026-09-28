#!/usr/bin/env python3
"""Plot TrackML hit x/y/z distributions before and after dataloader scaling."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VALID_SPLITS = ("train", "val", "test")
COORDS = ("x", "y", "z")
DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600.yaml")
DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/PAPERPLOTTING/plots/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load TrackML events using the same dataset config as the training dataloader "
            "and plot x/y/z hit-coordinate distributions before and after the "
            "coordinate scaling applied in TrackMLDataset.load_event()."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a TrackML YAML config.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
        help="Dataset split(s) to analyze.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events per split to process. Use -1 for all selected events.",
    )
    parser.add_argument(
        "--use-hit-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply split-specific hit-eval filtering from the config when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and CSV summaries are written.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read TrackML config files. Install it in the active environment and rerun the script."
        ) from exc

    with path.open() as f:
        return yaml.safe_load(f)


def _get_pyplot():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate hit-coordinate plots. Install it in the active environment and rerun the script."
        ) from exc

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _hist_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        center = float(values[0])
        return np.array([center - 0.5, center + 0.5], dtype=float)

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        pad = 0.5 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        if pad == 0.0:
            pad = 0.5
        return np.array([vmin - pad, vmax + pad], dtype=float)

    n_bins = int(np.clip(np.sqrt(values.size) * 2, 40, 200))
    return np.linspace(vmin, vmax, n_bins + 1)


def _resolve_num_events(configured_num_events: int, max_events: int) -> int:
    if max_events < 0:
        return configured_num_events
    if configured_num_events < 0:
        return max_events
    return min(configured_num_events, max_events)


def _build_dataset(config: dict, split: str, max_events: int, use_hit_eval: bool):
    from data import TrackMLDataset

    data_cfg = config["data"]
    num_events = _resolve_num_events(int(data_cfg[f"num_{split}"]), max_events)
    hit_eval_path = data_cfg.get(f"hit_eval_{split}") if use_hit_eval else None
    return TrackMLDataset(
        dirpath=data_cfg[f"{split}_dir"],
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        num_events=num_events,
        hit_volume_ids=data_cfg.get("hit_volume_ids"),
        feature_volume_ids=data_cfg.get("feature_volume_ids"),
        particle_min_pt=float(data_cfg.get("particle_min_pt", 1.0)),
        particle_max_abs_eta=float(data_cfg.get("particle_max_abs_eta", 2.5)),
        particle_min_num_hits=int(data_cfg.get("particle_min_num_hits", 3)),
        event_max_num_particles=int(data_cfg.get("event_max_num_particles", 1000)),
        strict_max_objects=bool(data_cfg.get("strict_max_objects", False)),
        hit_eval_path=hit_eval_path,
        dummy_data=bool(data_cfg.get("dummy_data", False)),
    )


def _collect_scaled_coords(dataset) -> dict[str, np.ndarray]:
    collected = {coord: [] for coord in COORDS}
    n_events = len(dataset)
    if n_events == 0:
        raise ValueError("No events available for the requested split and event limit.")

    for idx in range(n_events):
        hits, _ = dataset.load_event(idx)
        for coord in COORDS:
            collected[coord].append(hits[coord].to_numpy(copy=True))

        if (idx + 1) % 25 == 0 or (idx + 1) == n_events:
            print(f"Processed {idx + 1}/{n_events} events")

    return {coord: np.concatenate(chunks) for coord, chunks in collected.items()}


def _plot_split(
    split: str,
    scaled_coords: dict[str, np.ndarray],
    coord_scale: float,
    out_path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    plt = _get_pyplot()
    fig, axes = plt.subplots(len(COORDS), 2, figsize=(12, 11))
    summary: dict[tuple[str, str], dict[str, float]] = {}

    for row, coord in enumerate(COORDS):
        scaled = scaled_coords[coord]
        # load_event() has already applied the TrackML coordinate scale, so recover
        # the original values by dividing out the same factor for side-by-side plots.
        raw = scaled / coord_scale

        raw_edges = _hist_edges(raw)
        scaled_edges = raw_edges * coord_scale

        raw_ax = axes[row, 0]
        scaled_ax = axes[row, 1]

        raw_ax.hist(raw, bins=raw_edges, color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.35)
        scaled_ax.hist(scaled, bins=scaled_edges, color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.35)

        raw_ax.set_title(f"{coord}: raw")
        scaled_ax.set_title(f"{coord}: scaled")
        raw_ax.set_xlabel(f"{coord} before scaling")
        scaled_ax.set_xlabel(f"{coord} after scaling")
        raw_ax.set_ylabel("count")
        scaled_ax.set_ylabel("count")
        raw_ax.grid(alpha=0.25, linestyle="--")
        scaled_ax.grid(alpha=0.25, linestyle="--")

        summary[(coord, "raw")] = _stats(raw)
        summary[(coord, "scaled")] = _stats(scaled)

    fig.suptitle(f"{split}: hit-coordinate distributions before and after scaling (factor = {coord_scale:g})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _write_summary_csv(
    out_path: Path,
    split: str,
    config_path: Path,
    num_events: int,
    total_hits: int,
    coord_scale: float,
    summary: dict[tuple[str, str], dict[str, float]],
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["split", split])
        writer.writerow(["config", str(config_path)])
        writer.writerow(["num_events", num_events])
        writer.writerow(["total_hits", total_hits])
        writer.writerow(["coordinate_scale", coord_scale])
        writer.writerow([])
        writer.writerow(["coord", "variant", "min", "p05", "median", "mean", "std", "p95", "max"])
        for coord in COORDS:
            for variant in ("raw", "scaled"):
                stats = summary[(coord, variant)]
                writer.writerow(
                    [
                        coord,
                        variant,
                        stats["min"],
                        stats["p05"],
                        stats["median"],
                        stats["mean"],
                        stats["std"],
                        stats["p95"],
                        stats["max"],
                    ]
                )


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)

    from data import HIT_COORDINATE_SCALE

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_output_dir = args.output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading {split} split from {args.config}")
        dataset = _build_dataset(config=config, split=split, max_events=args.max_events, use_hit_eval=args.use_hit_eval)
        scaled_coords = _collect_scaled_coords(dataset)
        total_hits = int(sum(values.size for values in scaled_coords.values()) // len(COORDS))

        plot_path = split_output_dir / f"{split}_xyz_scaling.png"
        summary = _plot_split(
            split=split,
            scaled_coords=scaled_coords,
            coord_scale=HIT_COORDINATE_SCALE,
            out_path=plot_path,
        )

        summary_path = split_output_dir / f"{split}_xyz_scaling_summary.csv"
        _write_summary_csv(
            out_path=summary_path,
            split=split,
            config_path=args.config,
            num_events=len(dataset),
            total_hits=total_hits,
            coord_scale=HIT_COORDINATE_SCALE,
            summary=summary,
        )

        print(f"Saved plot to {plot_path}")
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
