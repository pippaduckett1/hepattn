#!/usr/bin/env python3
"""Paper-style plots of reconstructable tracks versus pre/post-filter event hit counts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml")
DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots/reconstructable_vs_hits/")
VALID_SPLITS = ("train", "val", "test")


@dataclass
class SplitCounts:
    split: str
    event_names: list[str]
    sample_ids: list[int]
    hits: np.ndarray
    particles: np.ndarray
    truth_track_hits: np.ndarray
    split_dir: Path
    hit_eval_path: Path | None


ALL_HITS = "all_hits"
VALID_TRACK_HITS = "valid_track_hits"
PRE_FILTER_HITS = "pre_filter_hits"
POST_FILTER_HITS = "post_filter_hits"


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-ready figure showing reconstructable tracks per event "
            "versus event hit counts for one or more TrackML splits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a TrackML YAML config.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["test"],
        help="Dataset split(s) to include as figure panels.",
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
        help="Use the split-specific hit-eval file from the config for the filtered figure.",
    )
    parser.add_argument(
        "--add-linear-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay a least-squares linear fit on each panel.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure and CSV files are written.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="reconstructable_vs_post_filter_hits",
        help="Filename stem used for output files.",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read TrackML config files. Install it in the active environment and rerun."
        ) from exc

    with path.open() as f:
        return yaml.safe_load(f)


def _get_pyplot():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the reconstructable-vs-hits figure. "
            "Install it in the active environment and rerun."
        ) from exc

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfc",
            "savefig.facecolor": "white",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _build_dataset(cfg: dict, split: str, use_hit_eval: bool):
    from data import TrackMLDataset

    data_cfg = cfg["data"]
    split_dir = Path(data_cfg[f"{split}_dir"])
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    split_num = int(data_cfg.get(f"num_{split}", -1))
    num_events = split_num if split_num > 0 else -1

    hit_eval_value = data_cfg.get(f"hit_eval_{split}")
    if use_hit_eval and not hit_eval_value:
        raise ValueError(
            f"Config does not define data.hit_eval_{split}, so post-hit-filter-hit counts cannot be computed for split '{split}'."
        )

    hit_eval_path = Path(hit_eval_value) if (use_hit_eval and hit_eval_value) else None
    if hit_eval_path is not None and not hit_eval_path.exists():
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


def _collect_counts(cfg: dict, split: str, max_events: int, use_hit_eval: bool, hit_count_mode: str) -> SplitCounts:
    dataset, split_dir, hit_eval_path = _build_dataset(cfg=cfg, split=split, use_hit_eval=use_hit_eval)

    n_total = len(dataset)
    n_use = n_total if max_events < 0 else min(max_events, n_total)
    if n_use <= 0:
        raise RuntimeError(f"No events selected for split '{split}'.")

    hits = np.zeros(n_use, dtype=np.int32)
    particles = np.zeros(n_use, dtype=np.int32)
    truth_track_hits_per_event: list[np.ndarray] = []
    event_names = dataset.event_names[:n_use]
    sample_ids = dataset.sample_ids[:n_use]

    print(f"[{split}] processing {n_use} events from {split_dir} with hit-eval {hit_eval_path} ({hit_count_mode})")
    for i in range(n_use):
        event_hits, event_particles = dataset.load_event(i)
        if "on_valid_particle" not in event_hits.columns:
            raise KeyError("Expected 'on_valid_particle' column on hits DataFrame.")

        valid_track_hits = event_hits.loc[event_hits["on_valid_particle"], "particle_id"].value_counts().to_numpy(dtype=np.int32)
        truth_track_hits_per_event.append(valid_track_hits)

        if hit_count_mode == ALL_HITS:
            hits[i] = len(event_hits)
        elif hit_count_mode == VALID_TRACK_HITS:
            hits[i] = int(valid_track_hits.sum())
        else:
            raise ValueError(f"Unknown hit_count_mode={hit_count_mode!r}")
        particles[i] = len(event_particles)
        if (i + 1) % 50 == 0 or (i + 1) == n_use:
            print(f"[{split}] processed {i + 1}/{n_use} events")

    return SplitCounts(
        split=split,
        event_names=event_names,
        sample_ids=sample_ids,
        hits=hits,
        particles=particles,
        truth_track_hits=np.concatenate(truth_track_hits_per_event),
        split_dir=split_dir,
        hit_eval_path=hit_eval_path,
    )


def _linear_fit(hits: np.ndarray, particles: np.ndarray) -> LinearFit | None:
    x = hits.astype(np.float64)
    y = particles.astype(np.float64)
    if x.size < 2 or np.unique(x).size < 2:
        return None

    slope, intercept = np.polyfit(x, y, deg=1)
    pred = intercept + slope * x
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LinearFit(slope=float(slope), intercept=float(intercept), r2=float(r2))


def _write_per_event_csv(
    out_path: Path,
    counts_by_split: list[SplitCounts],
    truth_counts_by_split: dict[str, SplitCounts] | None,
    hit_count_column: str,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["split", "event_index", "sample_id", "event_name", hit_count_column, "num_reconstructable_tracks"]
        if truth_counts_by_split is not None:
            header.extend(["num_truth_hits_before_filter", "num_truth_reconstructable_tracks"])
        writer.writerow(header)
        for split_counts in counts_by_split:
            truth_counts = None if truth_counts_by_split is None else truth_counts_by_split[split_counts.split]
            for idx, (hits, particles) in enumerate(zip(split_counts.hits, split_counts.particles, strict=True)):
                row = [
                    split_counts.split,
                    idx,
                    int(split_counts.sample_ids[idx]),
                    split_counts.event_names[idx],
                    int(hits),
                    int(particles),
                ]
                if truth_counts is not None:
                    row.extend([int(truth_counts.hits[idx]), int(truth_counts.particles[idx])])
                writer.writerow(row)


def _write_fit_summary_csv(out_path: Path, counts_by_split: list[SplitCounts], hit_mean_column: str) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "num_events",
                "split_dir",
                "hit_eval_path",
                hit_mean_column,
                "mean_reconstructable_tracks",
                "linear_fit_intercept",
                "linear_fit_slope",
                "linear_fit_r2",
            ]
        )
        for split_counts in counts_by_split:
            fit = _linear_fit(split_counts.hits, split_counts.particles)
            writer.writerow(
                [
                    split_counts.split,
                    len(split_counts.hits),
                    str(split_counts.split_dir),
                    str(split_counts.hit_eval_path),
                    float(np.mean(split_counts.hits)),
                    float(np.mean(split_counts.particles)),
                    "" if fit is None else fit.intercept,
                    "" if fit is None else fit.slope,
                    "" if fit is None else fit.r2,
                ]
            )


def _plot_panels(
    counts_by_split: list[SplitCounts],
    add_linear_fit: bool,
    output_base: Path,
    x_axis_label: str,
) -> None:
    plt = _get_pyplot()
    n_panels = len(counts_by_split)
    fig_width = max(6.2, 5.8 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 5.2), sharey=True, constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, split_counts in zip(axes, counts_by_split, strict=True):
        hits = split_counts.hits.astype(np.float64)
        particles = split_counts.particles.astype(np.float64)

        ax.scatter(
            hits,
            particles,
            s=5.0,
            marker=".",
            alpha=0.64,
            color="#1f77b4",
            edgecolors="none",
            zorder=2,
            rasterized=True,
        )

        fit = _linear_fit(hits, particles) if add_linear_fit else None
        if fit is not None:
            x_line = np.linspace(float(hits.min()), float(hits.max()) * 1.02, 256)
            y_line = fit.intercept + fit.slope * x_line
            ax.plot(
                x_line,
                y_line,
                color="#666666",
                linestyle="--",
                linewidth=0.85,
                zorder=4,
                alpha=0.78,
            )
            ax.text(
                0.08,
                0.90,
                f"Slope = {fit.slope:.4f}\n$R^2 = {fit.r2:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="#222222",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )

        x_min = float(hits.min())
        x_max = float(hits.max())
        y_min = float(particles.min())
        y_max = float(particles.max())
        x_pad = max(0.25, 0.015 * max(x_max - x_min, 1.0))
        y_pad = max(0.25, 0.02 * max(y_max - y_min, 1.0))
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

        ax.set_xlabel(x_axis_label)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Reconstructable tracks per event")

    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_plot_outputs(
    *,
    counts_by_split: list[SplitCounts],
    truth_counts_by_split: dict[str, SplitCounts],
    add_linear_fit: bool,
    output_base: Path,
    x_axis_label: str,
    hit_count_column: str,
    hit_mean_column: str,
) -> None:
    _plot_panels(
        counts_by_split=counts_by_split,
        add_linear_fit=add_linear_fit,
        output_base=output_base,
        x_axis_label=x_axis_label,
    )
    _write_per_event_csv(
        output_base.with_name(output_base.name + "_per_event.csv"),
        counts_by_split=counts_by_split,
        truth_counts_by_split=truth_counts_by_split,
        hit_count_column=hit_count_column,
    )
    _write_fit_summary_csv(
        output_base.with_name(output_base.name + "_fit_summary.csv"),
        counts_by_split=counts_by_split,
        hit_mean_column=hit_mean_column,
    )


def _validate_matching_events(reference: list[SplitCounts], candidate: list[SplitCounts]) -> dict[str, SplitCounts]:
    candidate_by_split = {split_counts.split: split_counts for split_counts in candidate}
    for ref in reference:
        other = candidate_by_split.get(ref.split)
        if other is None:
            raise RuntimeError(f"Missing split '{ref.split}' in truth-count collection.")
        if ref.event_names != other.event_names:
            raise RuntimeError(f"Event ordering mismatch for split '{ref.split}'.")
        if ref.sample_ids != other.sample_ids:
            raise RuntimeError(f"Sample-id mismatch for split '{ref.split}'.")
    return candidate_by_split


def _merge_metric(counts_by_split: list[SplitCounts], attribute: str) -> np.ndarray:
    return np.concatenate([getattr(split_counts, attribute) for split_counts in counts_by_split]).astype(np.float64)


def _print_summary_stats(
    *,
    pre_filter_counts_by_split: list[SplitCounts],
    post_filter_counts_by_split: list[SplitCounts],
) -> None:
    pre_filter_event_hits = _merge_metric(pre_filter_counts_by_split, "hits")
    post_filter_event_hits = _merge_metric(post_filter_counts_by_split, "hits")
    pre_filter_reconstructable_particles = _merge_metric(pre_filter_counts_by_split, "particles")
    post_filter_reconstructable_particles = _merge_metric(post_filter_counts_by_split, "particles")
    pre_filter_truth_track_hits = _merge_metric(pre_filter_counts_by_split, "truth_track_hits")
    post_filter_truth_track_hits = _merge_metric(post_filter_counts_by_split, "truth_track_hits")

    print(
        "Truth-track hits pre filter: "
        f"mean={np.mean(pre_filter_truth_track_hits):.3f}, std={np.std(pre_filter_truth_track_hits):.3f}"
    )
    print(
        "Truth-track hits post filter: "
        f"mean={np.mean(post_filter_truth_track_hits):.3f}, std={np.std(post_filter_truth_track_hits):.3f}"
    )
    print(
        "Reconstructable particles per event pre filter: "
        f"mean={np.mean(pre_filter_reconstructable_particles):.3f}, "
        f"std={np.std(pre_filter_reconstructable_particles):.3f}, "
        f"min={int(np.min(pre_filter_reconstructable_particles))}, "
        f"max={int(np.max(pre_filter_reconstructable_particles))}"
    )
    print(
        "Reconstructable particles per event post filter: "
        f"mean={np.mean(post_filter_reconstructable_particles):.3f}, "
        f"std={np.std(post_filter_reconstructable_particles):.3f}, "
        f"min={int(np.min(post_filter_reconstructable_particles))}, "
        f"max={int(np.max(post_filter_reconstructable_particles))}"
    )
    print(
        "Event hits pre filter over all selected events: "
        f"min={int(np.min(pre_filter_event_hits))}, max={int(np.max(pre_filter_event_hits))}"
    )
    print(
        "Event hits post filter over all selected events: "
        f"min={int(np.min(post_filter_event_hits))}, max={int(np.max(post_filter_event_hits))}"
    )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    cfg = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    post_filter_counts_by_split = [
        _collect_counts(
            cfg=cfg,
            split=split,
            max_events=args.max_events,
            use_hit_eval=args.use_hit_eval,
            hit_count_mode=ALL_HITS,
        )
        for split in args.splits
    ]

    pre_filter_counts_by_split = [
        _collect_counts(
            cfg=cfg,
            split=split,
            max_events=args.max_events,
            use_hit_eval=False,
            hit_count_mode=ALL_HITS,
        )
        for split in args.splits
    ]

    truth_counts_by_split_list = [
        _collect_counts(
            cfg=cfg,
            split=split,
            max_events=args.max_events,
            use_hit_eval=False,
            hit_count_mode=VALID_TRACK_HITS,
        )
        for split in args.splits
    ]
    truth_counts_by_split = _validate_matching_events(post_filter_counts_by_split, truth_counts_by_split_list)
    _validate_matching_events(post_filter_counts_by_split, pre_filter_counts_by_split)

    has_hit_eval = all(split_counts.hit_eval_path is not None for split_counts in post_filter_counts_by_split)
    post_filter_mode = POST_FILTER_HITS if has_hit_eval else ALL_HITS
    plot_configs = [
        {
            "mode": post_filter_mode,
            "counts_by_split": post_filter_counts_by_split,
            "output_base": output_dir / args.output_stem,
            "x_axis_label": "Hits after hit filtering" if has_hit_eval else "Hits per event",
            "hit_count_column": "num_hits_after_filter" if has_hit_eval else "num_hits",
            "hit_mean_column": "mean_hits_after_filter" if has_hit_eval else "mean_hits",
        },
        {
            "mode": PRE_FILTER_HITS,
            "counts_by_split": pre_filter_counts_by_split,
            "output_base": output_dir / f"{args.output_stem}_pre_filter",
            "x_axis_label": "Hits before hit filtering",
            "hit_count_column": "num_hits_before_filter",
            "hit_mean_column": "mean_hits_before_filter",
        },
    ]

    for plot_cfg in plot_configs:
        _write_plot_outputs(
            counts_by_split=plot_cfg["counts_by_split"],
            truth_counts_by_split=truth_counts_by_split,
            add_linear_fit=args.add_linear_fit,
            output_base=plot_cfg["output_base"],
            x_axis_label=plot_cfg["x_axis_label"],
            hit_count_column=plot_cfg["hit_count_column"],
            hit_mean_column=plot_cfg["hit_mean_column"],
        )

    truth_output_base = output_dir / f"{args.output_stem}_truth"
    _plot_panels(
        counts_by_split=truth_counts_by_split_list,
        add_linear_fit=args.add_linear_fit,
        output_base=truth_output_base,
        x_axis_label="Truth hits on reconstructable tracks",
    )
    _write_fit_summary_csv(
        truth_output_base.with_name(truth_output_base.name + "_fit_summary.csv"),
        counts_by_split=truth_counts_by_split_list,
        hit_mean_column="mean_truth_hits_on_reconstructable_tracks",
    )
    _print_summary_stats(
        pre_filter_counts_by_split=pre_filter_counts_by_split,
        post_filter_counts_by_split=post_filter_counts_by_split,
    )

    for plot_cfg in plot_configs:
        output_base = plot_cfg["output_base"]
        print(f"Saved {plot_cfg['mode']} figure to {output_base.with_suffix('.pdf')}")
        print(f"Saved {plot_cfg['mode']} figure to {output_base.with_suffix('.png')}")
        print(f"Saved {plot_cfg['mode']} per-event table to {output_base.with_name(output_base.name + '_per_event.csv')}")
        print(f"Saved {plot_cfg['mode']} fit summary to {output_base.with_name(output_base.name + '_fit_summary.csv')}")
    print(f"Saved figure to {truth_output_base.with_suffix('.pdf')}")
    print(f"Saved figure to {truth_output_base.with_suffix('.png')}")
    print(f"Saved fit summary to {truth_output_base.with_name(truth_output_base.name + '_fit_summary.csv')}")


if __name__ == "__main__":
    main()
