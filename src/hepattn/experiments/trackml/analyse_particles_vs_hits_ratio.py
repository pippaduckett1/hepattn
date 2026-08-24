#!/usr/bin/env python3
"""Analyze reconstructable-particle to hit ratios per event for TrackML configs.

This script uses ``TrackMLDataset.load_event()`` so event counts reflect the same
TrackML selection logic as the chosen config, without applying the fixed
``event_max_num_particles`` cap from ``__getitem__``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-.yaml")
DEFAULT_OUTPUT_DIR = Path("src/hepattn/experiments/trackml/analysis_plots/particles_vs_hits_ratio_900/")
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-event hit and reconstructable-particle counts, the particles/hits ratio, "
            "and upper-envelope recommendations for a dynamic max-particles cap."
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
        "--std-multiplier",
        type=float,
        default=2.0,
        help="Std multiplier used for one recommended upper-ratio line: mean + std_multiplier * std.",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=95.0,
        help="Quantile (in percent) used for one recommended upper-ratio line.",
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
        raise ModuleNotFoundError("PyYAML is required to read TrackML config files.") from exc

    with path.open() as f:
        return yaml.safe_load(f)


def _get_pyplot():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to generate analysis plots.") from exc

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def _build_dataset(cfg: dict, split: str, use_hit_eval: bool):
    from data import TrackMLDataset

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


def _hist_bins(values: np.ndarray) -> np.ndarray:
    n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
    return np.linspace(values.min(), values.max(), n_bins + 1)


def _plot_ratio_hist(ratio: np.ndarray, split: str, hit_label: str, out_path: Path) -> None:
    plt = _get_pyplot()
    stats = _stats(ratio)
    mean = stats["mean"]
    std = stats["std"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratio, bins=_hist_bins(ratio), alpha=0.85, color="tab:blue", edgecolor="black", linewidth=0.35)
    ax.axvline(mean, color="tab:red", linestyle="--", linewidth=1.5, label=f"mean = {mean:.4f}")
    if std > 0:
        ax.axvspan(mean - std, mean + std, color="tab:red", alpha=0.12, label=f"+/- 1 std = {std:.4f}")
    ax.set_xlabel(f"particles / {hit_label}")
    ax.set_ylabel("count")
    ax.set_title(f"{split}: reconstructable-particles to {hit_label} ratio per event")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter_with_ratio_lines(
    hits: np.ndarray,
    particles: np.ndarray,
    ratio_lines: dict[str, float],
    split: str,
    hit_axis_label: str,
    out_path: Path,
) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hits, particles, s=10, alpha=0.45, color="tab:blue", label="events")

    hits_line = np.linspace(0.0, float(hits.max()) * 1.02, 256)
    colors = ["tab:red", "tab:orange", "tab:green"]
    for (label, ratio), color in zip(ratio_lines.items(), colors, strict=False):
        ax.plot(hits_line, ratio * hits_line, color=color, linewidth=2.0, label=f"{label}: particles = {ratio:.4f} * {hit_axis_label}")

    ax.set_xlabel(f"{hit_axis_label} per event")
    ax.set_ylabel("reconstructable particles per event")
    ax.set_title(f"{split}: particles vs {hit_axis_label} per event")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_ratio_vs_hits(hits: np.ndarray, ratio: np.ndarray, split: str, hit_axis_label: str, out_path: Path) -> None:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hits, ratio, s=10, alpha=0.45, color="tab:purple")
    ax.set_xlabel(f"{hit_axis_label} per event")
    ax.set_ylabel(f"particles / {hit_axis_label}")
    ax.set_title(f"{split}: ratio vs {hit_axis_label} per event")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _linear_fit(particles: np.ndarray, hits: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(hits.astype(np.float64), particles.astype(np.float64), deg=1)
    pred = intercept + slope * hits.astype(np.float64)
    residuals = particles.astype(np.float64) - pred
    ss_res = float(np.square(residuals).sum())
    ss_tot = float(np.square(particles.astype(np.float64) - particles.astype(np.float64).mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_std": float(np.std(residuals)),
        "r2": float(r2),
    }


def _affine_rule_stats(particles: np.ndarray, hits: np.ndarray) -> dict[str, float]:
    fit = _linear_fit(particles=particles, hits=hits)
    pred = fit["intercept"] + fit["slope"] * hits.astype(np.float64)
    residuals = particles.astype(np.float64) - pred
    positive_residuals = np.clip(residuals, a_min=0.0, a_max=None)

    fit["residual_mean"] = float(np.mean(residuals))
    fit["positive_residual_p95"] = float(np.percentile(positive_residuals, 95))
    fit["positive_residual_p99"] = float(np.percentile(positive_residuals, 99))
    fit["positive_residual_max"] = float(np.max(positive_residuals))
    return fit


def _affine_cap(hits: np.ndarray, intercept: float, slope: float, margin: float) -> np.ndarray:
    return np.ceil(intercept + slope * hits.astype(np.float64) + margin).astype(np.int32)


def _affine_coverage(particles: np.ndarray, hits: np.ndarray, intercept: float, slope: float, margin: float) -> float:
    return float(np.mean(particles <= _affine_cap(hits, intercept=intercept, slope=slope, margin=margin)))


def _plot_particles_vs_hits_linear_fit(
    hits: np.ndarray,
    particles: np.ndarray,
    split: str,
    hit_axis_label: str,
    out_path: Path,
) -> dict[str, float]:
    plt = _get_pyplot()
    fit = _affine_rule_stats(particles=particles, hits=hits)

    x_line = np.linspace(float(hits.min()), float(hits.max()) * 1.02, 256)
    y_line = fit["intercept"] + fit["slope"] * x_line
    y_err = np.full_like(x_line, fit["residual_std"])
    y_line_p95 = y_line + fit["positive_residual_p95"]
    y_line_p99 = y_line + fit["positive_residual_p99"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hits, particles, s=10, alpha=0.35, color="tab:blue", label="events")
    ax.plot(
        x_line,
        y_line,
        color="tab:red",
        linewidth=2.2,
        label=f"fit: particles = {fit['intercept']:.2f} + {fit['slope']:.4f} * {hit_axis_label}",
    )
    ax.fill_between(
        x_line,
        y_line - y_err,
        y_line + y_err,
        color="tab:red",
        alpha=0.14,
        label=f"fit +/- 1 residual std = {fit['residual_std']:.2f}",
    )
    ax.plot(
        x_line,
        y_line_p95,
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label=f"fit + positive residual p95 = {fit['positive_residual_p95']:.2f}",
    )
    ax.plot(
        x_line,
        y_line_p99,
        color="tab:green",
        linewidth=1.8,
        linestyle="--",
        label=f"fit + positive residual p99 = {fit['positive_residual_p99']:.2f}",
    )

    sample_idx = np.linspace(0, len(x_line) - 1, num=8, dtype=int)
    ax.errorbar(
        x_line[sample_idx],
        y_line[sample_idx],
        yerr=y_err[sample_idx],
        fmt="none",
        ecolor="tab:red",
        elinewidth=1.2,
        capsize=3,
        alpha=0.8,
    )

    ax.set_xlabel(f"{hit_axis_label} per event")
    ax.set_ylabel("reconstructable particles per event")
    ax.set_title(f"{split}: particles vs {hit_axis_label} with linear fit")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(title=f"R^2 = {fit['r2']:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return fit


def _coverage(particles: np.ndarray, hits: np.ndarray, ratio: float) -> float:
    dynamic_cap = np.ceil(ratio * hits).astype(np.int32)
    return float(np.mean(particles <= dynamic_cap))


def _write_per_event_csv(
    out_path: Path,
    split: str,
    event_names: list[str],
    sample_ids: list[int],
    hit_count_column: str,
    hits: np.ndarray,
    particles: np.ndarray,
    ratio: np.ndarray,
    ratio_upper_std: float,
    ratio_upper_quantile: float,
    affine_fit: dict[str, float],
) -> None:
    dynamic_cap_std = np.ceil(ratio_upper_std * hits).astype(np.int32)
    dynamic_cap_quantile = np.ceil(ratio_upper_quantile * hits).astype(np.int32)
    dynamic_cap_affine_p95 = _affine_cap(
        hits,
        intercept=affine_fit["intercept"],
        slope=affine_fit["slope"],
        margin=affine_fit["positive_residual_p95"],
    )
    dynamic_cap_affine_p99 = _affine_cap(
        hits,
        intercept=affine_fit["intercept"],
        slope=affine_fit["slope"],
        margin=affine_fit["positive_residual_p99"],
    )

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "event_index",
                "sample_id",
                "event_name",
                hit_count_column,
                "num_particles",
                "particles_per_hit",
                "hits_per_particle",
                "dynamic_cap_mean_plus_nstd",
                "dynamic_cap_quantile",
                "dynamic_cap_affine_p95_positive_residual",
                "dynamic_cap_affine_p99_positive_residual",
            ]
        )
        for i in range(len(hits)):
            writer.writerow(
                [
                    split,
                    i,
                    int(sample_ids[i]),
                    event_names[i],
                    int(hits[i]),
                    int(particles[i]),
                    float(ratio[i]),
                    float(hits[i] / max(float(particles[i]), 1.0)),
                    int(dynamic_cap_std[i]),
                    int(dynamic_cap_quantile[i]),
                    int(dynamic_cap_affine_p95[i]),
                    int(dynamic_cap_affine_p99[i]),
                ]
            )


def _write_summary_csv(
    out_path: Path,
    split: str,
    config_path: Path,
    split_dir: Path,
    hit_eval_path: Path | None,
    current_cap: int,
    hit_count_metric_prefix: str,
    hits: np.ndarray,
    particles: np.ndarray,
    ratio: np.ndarray,
    ratio_upper_std: float,
    ratio_upper_quantile: float,
    std_multiplier: float,
    upper_quantile: float,
    affine_fit: dict[str, float],
) -> None:
    hit_stats = _stats(hits.astype(np.float64))
    particle_stats = _stats(particles.astype(np.float64))
    ratio_stats = _stats(ratio)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["split", split])
        writer.writerow(["config", str(config_path)])
        writer.writerow(["split_dir", str(split_dir)])
        writer.writerow(["hit_eval_path", "" if hit_eval_path is None else str(hit_eval_path)])
        writer.writerow(["num_events", len(hits)])
        writer.writerow(["current_event_max_num_particles", int(current_cap)])
        writer.writerow(["frac_events_over_current_cap", float(np.mean(particles > current_cap))])
        writer.writerow([])
        writer.writerow([f"{hit_count_metric_prefix}_min", hit_stats["min"]])
        writer.writerow([f"{hit_count_metric_prefix}_p05", hit_stats["p05"]])
        writer.writerow([f"{hit_count_metric_prefix}_median", hit_stats["median"]])
        writer.writerow([f"{hit_count_metric_prefix}_mean", hit_stats["mean"]])
        writer.writerow([f"{hit_count_metric_prefix}_std", hit_stats["std"]])
        writer.writerow([f"{hit_count_metric_prefix}_p95", hit_stats["p95"]])
        writer.writerow([f"{hit_count_metric_prefix}_p99", hit_stats["p99"]])
        writer.writerow([f"{hit_count_metric_prefix}_max", hit_stats["max"]])
        writer.writerow([])
        writer.writerow(["particles_min", particle_stats["min"]])
        writer.writerow(["particles_p05", particle_stats["p05"]])
        writer.writerow(["particles_median", particle_stats["median"]])
        writer.writerow(["particles_mean", particle_stats["mean"]])
        writer.writerow(["particles_std", particle_stats["std"]])
        writer.writerow(["particles_p95", particle_stats["p95"]])
        writer.writerow(["particles_p99", particle_stats["p99"]])
        writer.writerow(["particles_max", particle_stats["max"]])
        writer.writerow([])
        writer.writerow(["particles_per_hit_min", ratio_stats["min"]])
        writer.writerow(["particles_per_hit_p05", ratio_stats["p05"]])
        writer.writerow(["particles_per_hit_median", ratio_stats["median"]])
        writer.writerow(["particles_per_hit_mean", ratio_stats["mean"]])
        writer.writerow(["particles_per_hit_std", ratio_stats["std"]])
        writer.writerow(["particles_per_hit_p95", ratio_stats["p95"]])
        writer.writerow(["particles_per_hit_p99", ratio_stats["p99"]])
        writer.writerow(["particles_per_hit_max", ratio_stats["max"]])
        writer.writerow([])
        writer.writerow([f"recommended_ratio_mean_plus_{std_multiplier:.2f}std", ratio_upper_std])
        writer.writerow([f"coverage_mean_plus_{std_multiplier:.2f}std", _coverage(particles, hits, ratio_upper_std)])
        writer.writerow([f"recommended_ratio_p{upper_quantile:.0f}", ratio_upper_quantile])
        writer.writerow([f"coverage_p{upper_quantile:.0f}", _coverage(particles, hits, ratio_upper_quantile)])
        writer.writerow(["recommended_ratio_max", ratio_stats["max"]])
        writer.writerow(["coverage_max", _coverage(particles, hits, ratio_stats["max"])])
        writer.writerow([])
        writer.writerow(["linear_fit_intercept", affine_fit["intercept"]])
        writer.writerow(["linear_fit_slope", affine_fit["slope"]])
        writer.writerow(["linear_fit_residual_std", affine_fit["residual_std"]])
        writer.writerow(["linear_fit_residual_mean", affine_fit["residual_mean"]])
        writer.writerow(["linear_fit_r2", affine_fit["r2"]])
        writer.writerow(["positive_residual_p95", affine_fit["positive_residual_p95"]])
        writer.writerow(["positive_residual_p99", affine_fit["positive_residual_p99"]])
        writer.writerow(["positive_residual_max", affine_fit["positive_residual_max"]])
        writer.writerow([])
        writer.writerow(["recommended_affine_intercept", affine_fit["intercept"]])
        writer.writerow(["recommended_affine_slope", affine_fit["slope"]])
        writer.writerow(["recommended_affine_margin_p95_positive_residual", affine_fit["positive_residual_p95"]])
        writer.writerow([
            "coverage_affine_p95_positive_residual",
            _affine_coverage(
                particles,
                hits,
                intercept=affine_fit["intercept"],
                slope=affine_fit["slope"],
                margin=affine_fit["positive_residual_p95"],
            ),
        ])
        writer.writerow(["recommended_affine_margin_p99_positive_residual", affine_fit["positive_residual_p99"]])
        writer.writerow([
            "coverage_affine_p99_positive_residual",
            _affine_coverage(
                particles,
                hits,
                intercept=affine_fit["intercept"],
                slope=affine_fit["slope"],
                margin=affine_fit["positive_residual_p99"],
            ),
        ])


def _analyze_split(
    cfg: dict,
    config_path: Path,
    split: str,
    max_events: int,
    use_hit_eval: bool,
    output_dir: Path,
    std_multiplier: float,
    upper_quantile: float,
) -> None:
    dataset, split_dir, hit_eval_path = _build_dataset(cfg, split=split, use_hit_eval=use_hit_eval)
    hit_axis_label = "post-hit-filter hits" if hit_eval_path is not None else "hits"
    hit_count_column = "num_post_hit_filter_hits" if hit_eval_path is not None else "num_hits"
    hit_count_metric_prefix = "post_hit_filter_hits" if hit_eval_path is not None else "hits"

    n_total = len(dataset)
    n_use = n_total if max_events < 0 else min(max_events, n_total)
    if n_use <= 0:
        raise RuntimeError(f"No events selected for split '{split}'.")

    hits = np.zeros(n_use, dtype=np.int32)
    particles = np.zeros(n_use, dtype=np.int32)
    event_names = dataset.event_names[:n_use]
    sample_ids = dataset.sample_ids[:n_use]

    print(
        f"[{split}] processing {n_use} events from {split_dir}"
        + (f" with hit-eval {hit_eval_path}" if hit_eval_path is not None else " without hit-eval filtering")
    )

    for i in range(n_use):
        event_hits, event_particles = dataset.load_event(i)
        hits[i] = len(event_hits)
        particles[i] = len(event_particles)
        if (i + 1) % 50 == 0 or (i + 1) == n_use:
            print(f"[{split}] processed {i + 1}/{n_use} events")

    ratio = particles.astype(np.float64) / np.maximum(hits.astype(np.float64), 1.0)
    ratio_stats = _stats(ratio)
    ratio_upper_std = ratio_stats["mean"] + std_multiplier * ratio_stats["std"]
    ratio_upper_quantile = float(np.percentile(ratio, upper_quantile))

    print(
        f"[{split}] particles/{hit_axis_label} ratio: mean={ratio_stats['mean']:.4f}, std={ratio_stats['std']:.4f}, "
        f"p95={ratio_stats['p95']:.4f}, p99={ratio_stats['p99']:.4f}, max={ratio_stats['max']:.4f}"
    )
    print(
        f"[{split}] recommended upper ratios: mean+{std_multiplier:.2f}std={ratio_upper_std:.4f} "
        f"(coverage={_coverage(particles, hits, ratio_upper_std):.3f}), "
        f"p{upper_quantile:.0f}={ratio_upper_quantile:.4f} "
        f"(coverage={_coverage(particles, hits, ratio_upper_quantile):.3f})"
    )
    affine_fit = _affine_rule_stats(particles=particles, hits=hits)
    print(
        f"[{split}] linear fit: particles = {affine_fit['intercept']:.2f} + {affine_fit['slope']:.4f} * {hit_axis_label}, "
        f"residual_std={affine_fit['residual_std']:.2f}, r2={affine_fit['r2']:.4f}"
    )
    print(
        f"[{split}] affine margins from positive residuals: "
        f"p95={affine_fit['positive_residual_p95']:.2f} "
        f"(coverage={_affine_coverage(particles, hits, intercept=affine_fit['intercept'], slope=affine_fit['slope'], margin=affine_fit['positive_residual_p95']):.3f}), "
        f"p99={affine_fit['positive_residual_p99']:.2f} "
        f"(coverage={_affine_coverage(particles, hits, intercept=affine_fit['intercept'], slope=affine_fit['slope'], margin=affine_fit['positive_residual_p99']):.3f})"
    )

    _plot_ratio_hist(ratio, split=split, hit_label=hit_axis_label, out_path=output_dir / f"{split}_particles_per_hit_ratio_hist.png")
    _plot_ratio_vs_hits(hits, ratio, split=split, hit_axis_label=hit_axis_label, out_path=output_dir / f"{split}_particles_per_hit_ratio_vs_hits.png")
    _plot_scatter_with_ratio_lines(
        hits=hits.astype(np.float64),
        particles=particles.astype(np.float64),
        ratio_lines={
            f"mean+{std_multiplier:.1f}std": ratio_upper_std,
            f"p{upper_quantile:.0f}": ratio_upper_quantile,
            "max": ratio_stats["max"],
        },
        split=split,
        hit_axis_label=hit_axis_label,
        out_path=output_dir / f"{split}_particles_vs_hits_scatter.png",
    )
    affine_fit = _plot_particles_vs_hits_linear_fit(
        hits=hits.astype(np.float64),
        particles=particles.astype(np.float64),
        split=split,
        hit_axis_label=hit_axis_label,
        out_path=output_dir / f"{split}_particles_vs_hits_linear_fit.png",
    )
    _write_per_event_csv(
        out_path=output_dir / f"{split}_particles_vs_hits_ratio_per_event.csv",
        split=split,
        event_names=event_names,
        sample_ids=sample_ids,
        hit_count_column=hit_count_column,
        hits=hits,
        particles=particles,
        ratio=ratio,
        ratio_upper_std=ratio_upper_std,
        ratio_upper_quantile=ratio_upper_quantile,
        affine_fit=affine_fit,
    )
    _write_summary_csv(
        out_path=output_dir / f"{split}_particles_vs_hits_ratio_summary.csv",
        split=split,
        config_path=config_path,
        split_dir=split_dir,
        hit_eval_path=hit_eval_path,
        current_cap=int(cfg["data"]["event_max_num_particles"]),
        hit_count_metric_prefix=hit_count_metric_prefix,
        hits=hits,
        particles=particles,
        ratio=ratio,
        ratio_upper_std=ratio_upper_std,
        ratio_upper_quantile=ratio_upper_quantile,
        std_multiplier=std_multiplier,
        upper_quantile=upper_quantile,
        affine_fit=affine_fit,
    )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    cfg = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        _analyze_split(
            cfg=cfg,
            config_path=args.config,
            split=split,
            max_events=args.max_events,
            use_hit_eval=args.use_hit_eval,
            output_dir=output_dir,
            std_multiplier=args.std_multiplier,
            upper_quantile=args.upper_quantile,
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
