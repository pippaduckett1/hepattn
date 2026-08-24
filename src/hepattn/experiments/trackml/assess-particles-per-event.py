#!/usr/bin/env python3
"""Assess reconstructable-particle counts per event from a TrackML config."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-.yaml")
# DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/reconstructable_parts/")
# VALID_SPLITS = ("train", "val", "test")


DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600.yaml")
DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/reconstructable_parts_600/")
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reconstructable-particle counts per event using the same "
            "TrackMLDataset selection logic as the chosen config."
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
        help="Apply the split-specific hit-eval filtering from the config when available.",
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


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _get_pyplot():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate reconstructable-particle plots. "
            "Install it in the active environment and rerun the script."
        ) from exc

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _hist_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        center = float(values[0])
        return np.array([center - 0.5, center + 0.5], dtype=float)
    if np.all(values == values.astype(np.int64)) and (values.max() - values.min()) <= 160:
        return np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
    n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
    return np.linspace(values.min(), values.max(), n_bins + 1)


def _plot_hist(values: np.ndarray, split: str, out_path: Path) -> None:
    plt = _get_pyplot()
    stats = _stats(values)
    mean = stats["mean"]
    std = stats["std"]
    bins = _hist_bins(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, alpha=0.85, color="tab:blue", edgecolor="black", linewidth=0.35)
    ax.axvline(mean, color="tab:red", linestyle="--", linewidth=1.5, label=f"mean = {mean:.2f}")
    if std > 0:
        ax.axvspan(mean - std, mean + std, color="tab:red", alpha=0.12, label=f"+/- 1 std = {std:.2f}")
    ax.set_xlabel("reconstructable particles per event")
    ax.set_ylabel("count")
    ax.set_title(f"{split}: distribution of reconstructable particles per event")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overlay(counts_by_split: dict[str, np.ndarray], out_path: Path) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))
    for split, values in counts_by_split.items():
        stats = _stats(values)
        ax.hist(
            values,
            bins=_hist_bins(values),
            histtype="step",
            linewidth=1.8,
            label=f"{split} (mean={stats['mean']:.2f}, std={stats['std']:.2f})",
        )

    ax.set_xlabel("reconstructable particles per event")
    ax.set_ylabel("count")
    ax.set_title("Reconstructable particles per event by split")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_per_event_csv(
    out_path: Path,
    split: str,
    event_names: list[str],
    sample_ids: list[int],
    pre_particle_counts: np.ndarray,
    post_particle_counts: np.ndarray,
    pre_hit_counts: np.ndarray,
    post_hit_counts: np.ndarray,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "event_index",
                "sample_id",
                "event_name",
                "reconstructable_particles_pre_filter",
                "reconstructable_particles_post_filter",
                "hits_pre_filter",
                "hits_post_filter",
            ]
        )
        for i in range(len(post_particle_counts)):
            writer.writerow(
                [
                    split,
                    i,
                    int(sample_ids[i]),
                    event_names[i],
                    int(pre_particle_counts[i]),
                    int(post_particle_counts[i]),
                    int(pre_hit_counts[i]),
                    int(post_hit_counts[i]),
                ]
            )


def _write_summary_csv(
    out_path: Path,
    split: str,
    config_path: Path,
    split_dir: Path,
    hit_eval_path: Path | None,
    n_events: int,
    pre_particle_stats: dict[str, float],
    post_particle_stats: dict[str, float],
    pre_hit_stats: dict[str, float],
    post_hit_stats: dict[str, float],
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["split", split])
        writer.writerow(["config", str(config_path)])
        writer.writerow(["split_dir", str(split_dir)])
        writer.writerow(["hit_eval_path", "" if hit_eval_path is None else str(hit_eval_path)])
        writer.writerow(["num_events", n_events])
        writer.writerow([])
        writer.writerow(["reconstructable_particles_pre_filter_min", pre_particle_stats["min"]])
        writer.writerow(["reconstructable_particles_pre_filter_p05", pre_particle_stats["p05"]])
        writer.writerow(["reconstructable_particles_pre_filter_median", pre_particle_stats["median"]])
        writer.writerow(["reconstructable_particles_pre_filter_mean", pre_particle_stats["mean"]])
        writer.writerow(["reconstructable_particles_pre_filter_std", pre_particle_stats["std"]])
        writer.writerow(["reconstructable_particles_pre_filter_p95", pre_particle_stats["p95"]])
        writer.writerow(["reconstructable_particles_pre_filter_max", pre_particle_stats["max"]])
        writer.writerow(["reconstructable_particles_post_filter_min", post_particle_stats["min"]])
        writer.writerow(["reconstructable_particles_post_filter_p05", post_particle_stats["p05"]])
        writer.writerow(["reconstructable_particles_post_filter_median", post_particle_stats["median"]])
        writer.writerow(["reconstructable_particles_post_filter_mean", post_particle_stats["mean"]])
        writer.writerow(["reconstructable_particles_post_filter_std", post_particle_stats["std"]])
        writer.writerow(["reconstructable_particles_post_filter_p95", post_particle_stats["p95"]])
        writer.writerow(["reconstructable_particles_post_filter_max", post_particle_stats["max"]])
        writer.writerow(["hits_pre_filter_min", pre_hit_stats["min"]])
        writer.writerow(["hits_pre_filter_p05", pre_hit_stats["p05"]])
        writer.writerow(["hits_pre_filter_median", pre_hit_stats["median"]])
        writer.writerow(["hits_pre_filter_mean", pre_hit_stats["mean"]])
        writer.writerow(["hits_pre_filter_std", pre_hit_stats["std"]])
        writer.writerow(["hits_pre_filter_p95", pre_hit_stats["p95"]])
        writer.writerow(["hits_pre_filter_max", pre_hit_stats["max"]])
        writer.writerow(["hits_post_filter_min", post_hit_stats["min"]])
        writer.writerow(["hits_post_filter_p05", post_hit_stats["p05"]])
        writer.writerow(["hits_post_filter_median", post_hit_stats["median"]])
        writer.writerow(["hits_post_filter_mean", post_hit_stats["mean"]])
        writer.writerow(["hits_post_filter_std", post_hit_stats["std"]])
        writer.writerow(["hits_post_filter_p95", post_hit_stats["p95"]])
        writer.writerow(["hits_post_filter_max", post_hit_stats["max"]])


def _build_dataset(cfg: dict, split: str, use_hit_eval: bool):
    from hepattn.experiments.trackml.data import TrackMLDataset

    data_cfg = cfg["data"]
    split_dir = Path(data_cfg[f"{split}_dir"])
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    split_num = int(data_cfg.get(f"num_{split}", -1))
    num_events = split_num if split_num > 0 else -1

    hit_eval_path: Path | None = None
    hit_eval_value = data_cfg.get(f"hit_eval_{split}")
    if use_hit_eval and hit_eval_value:
        hit_eval_path = Path(hit_eval_value)
        if not hit_eval_path.exists():
            raise FileNotFoundError(f"Configured hit-eval file does not exist for split '{split}': {hit_eval_path}")

    dataset = TrackMLDataset(
        dirpath=str(split_dir),
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        num_events=num_events,
        hit_volume_ids=data_cfg.get("hit_volume_ids"),
        feature_volume_ids=data_cfg.get("feature_volume_ids"),
        particle_min_pt=data_cfg["particle_min_pt"],
        particle_max_abs_eta=data_cfg["particle_max_abs_eta"],
        particle_min_num_hits=data_cfg["particle_min_num_hits"],
        event_max_num_particles=data_cfg["event_max_num_particles"],
        strict_max_objects=data_cfg.get("strict_max_objects", False),
        hit_eval_path=None if hit_eval_path is None else str(hit_eval_path),
        hit_filter_threshold=data_cfg.get("hit_filter_threshold", 0.1),
    )
    return dataset, split_dir, hit_eval_path


def _analyze_split(cfg: dict, config_path: Path, split: str, max_events: int, use_hit_eval: bool, output_dir: Path) -> np.ndarray:
    pre_dataset, split_dir, _pre_hit_eval_path = _build_dataset(cfg, split=split, use_hit_eval=False)
    post_dataset, _post_split_dir, hit_eval_path = _build_dataset(cfg, split=split, use_hit_eval=use_hit_eval)

    if pre_dataset.event_names != post_dataset.event_names or pre_dataset.sample_ids != post_dataset.sample_ids:
        raise RuntimeError(f"Pre/post datasets are not aligned for split '{split}'.")

    n_total = len(post_dataset)
    n_use = n_total if max_events < 0 else min(max_events, n_total)
    if n_use <= 0:
        raise RuntimeError(f"No events selected for split '{split}'.")

    pre_particle_counts = np.zeros(n_use, dtype=np.int32)
    post_particle_counts = np.zeros(n_use, dtype=np.int32)
    pre_hit_counts = np.zeros(n_use, dtype=np.int32)
    post_hit_counts = np.zeros(n_use, dtype=np.int32)
    event_names = post_dataset.event_names[:n_use]
    sample_ids = post_dataset.sample_ids[:n_use]

    print(
        f"[{split}] processing {n_use} events from {split_dir}"
        + (f" with hit-eval {hit_eval_path}" if hit_eval_path is not None else " without hit-eval filtering")
    )

    for i in range(n_use):
        pre_hits, pre_particles = pre_dataset.load_event(i)
        post_hits, post_particles = post_dataset.load_event(i)
        pre_particle_counts[i] = len(pre_particles)
        post_particle_counts[i] = len(post_particles)
        pre_hit_counts[i] = len(pre_hits)
        post_hit_counts[i] = len(post_hits)
        if (i + 1) % 50 == 0 or (i + 1) == n_use:
            print(f"[{split}] processed {i + 1}/{n_use} events")

    pre_particle_stats = _stats(pre_particle_counts.astype(np.float64))
    post_particle_stats = _stats(post_particle_counts.astype(np.float64))
    pre_hit_stats = _stats(pre_hit_counts.astype(np.float64))
    post_hit_stats = _stats(post_hit_counts.astype(np.float64))
    print(
        f"[{split}] particles/event pre: mean={pre_particle_stats['mean']:.2f}, std={pre_particle_stats['std']:.2f}, "
        f"max={pre_particle_stats['max']:.0f}; post: mean={post_particle_stats['mean']:.2f}, "
        f"std={post_particle_stats['std']:.2f}, max={post_particle_stats['max']:.0f}"
    )
    print(
        f"[{split}] hits/event pre: mean={pre_hit_stats['mean']:.2f}, std={pre_hit_stats['std']:.2f}, "
        f"max={pre_hit_stats['max']:.0f}; post: mean={post_hit_stats['mean']:.2f}, "
        f"std={post_hit_stats['std']:.2f}, max={post_hit_stats['max']:.0f}"
    )

    _plot_hist(
        values=post_particle_counts.astype(np.float64),
        split=split,
        out_path=output_dir / f"{split}_reconstructable_particles_per_event_hist.png",
    )
    _write_per_event_csv(
        out_path=output_dir / f"{split}_reconstructable_particles_per_event.csv",
        split=split,
        event_names=event_names,
        sample_ids=sample_ids,
        pre_particle_counts=pre_particle_counts,
        post_particle_counts=post_particle_counts,
        pre_hit_counts=pre_hit_counts,
        post_hit_counts=post_hit_counts,
    )
    _write_summary_csv(
        out_path=output_dir / f"{split}_reconstructable_particles_summary.csv",
        split=split,
        config_path=config_path,
        split_dir=split_dir,
        hit_eval_path=hit_eval_path,
        n_events=n_use,
        pre_particle_stats=pre_particle_stats,
        post_particle_stats=post_particle_stats,
        pre_hit_stats=pre_hit_stats,
        post_hit_stats=post_hit_stats,
    )
    return post_particle_counts


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    cfg = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_by_split: dict[str, np.ndarray] = {}
    for split in args.splits:
        counts_by_split[split] = _analyze_split(
            cfg=cfg,
            config_path=args.config,
            split=split,
            max_events=args.max_events,
            use_hit_eval=args.use_hit_eval,
            output_dir=output_dir,
        )

    if len(counts_by_split) > 1:
        _plot_overlay(
            counts_by_split=counts_by_split,
            out_path=output_dir / "reconstructable_particles_per_event_overlay.png",
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
