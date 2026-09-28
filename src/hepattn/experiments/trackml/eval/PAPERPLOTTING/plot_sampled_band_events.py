#!/usr/bin/env python3
"""Per-event truth/noise comparisons for randomly sampled events from each band."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[4]
THIS_DIR = Path(__file__).resolve().parent
for _path in (SRC_ROOT, THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from explain_bands import (  # noqa: E402
    ALL_HITS,
    BAND_COLORS,
    BAND_NAMES,
    DEFAULT_CONFIG,
    VALID_BAND_CLUSTERING,
    VALID_SPLITS,
    _assign_event_bands,
    _collect_counts,
    _get_pyplot,
    _load_raw_trackml_event,
    _read_yaml,
)


DEFAULT_OUTPUT_DIR = Path(
    "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots/sampled_band_events/"
)


@dataclass
class EventViews:
    hits_raw: pd.DataFrame
    particles_raw: pd.DataFrame
    truth_particles: pd.DataFrame
    truth_hits: pd.DataFrame
    lowpt_particles: pd.DataFrame
    lowpt_hits: pd.DataFrame
    noise_hits: pd.DataFrame
    metrics: dict[str, float | int | str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample per-event comparisons from lower/upper post-filter bands, "
            "including truth, low-pT, noise, and event-level summary distributions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TrackML YAML config.")
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=["test"], help="Dataset split(s) to analyze.")
    parser.add_argument(
        "--band-assignments-csv",
        type=Path,
        default=None,
        help="Optional precomputed band assignments CSV from plot_reconstructable_vs_post_filter_hits.py.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events per split to use when resolving assignments (-1 = all).",
    )
    parser.add_argument(
        "--band-clustering",
        choices=VALID_BAND_CLUSTERING,
        default="parallel-lines",
        help="Band splitting method when assignments are recomputed from the config.",
    )
    parser.add_argument(
        "--sample-events-per-band",
        type=int,
        default=20,
        help="Number of random events to sample from each band for per-event figures.",
    )
    parser.add_argument("--sample-seed", type=int, default=12345, help="RNG seed for event sampling.")
    parser.add_argument(
        "--lowpt-threshold",
        type=float,
        default=0.9,
        help="Low-pT threshold [GeV] used for per-event low-pT particle and hit summaries.",
    )
    parser.add_argument(
        "--max-tracks-on-xy",
        type=int,
        default=24,
        help="Maximum number of reconstructable truth tracks to overlay on the x-y hit map.",
    )
    parser.add_argument(
        "--hist-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize histogram panels to density instead of raw event counts.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for plots and CSV summaries.")
    parser.add_argument("--output-stem", type=str, default="sampled_band_events", help="Filename stem for outputs.")
    return parser.parse_args()


def _continuous_bins(values: np.ndarray, *, n_bins: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.min(values))
        hi = float(np.max(values))
    if np.isclose(lo, hi):
        delta = max(1e-6, abs(lo) * 0.05 + 1e-6)
        lo -= delta
        hi += delta
    return np.linspace(lo, hi, n_bins + 1)


def _count_bins(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.arange(-0.5, 1.5, 1.0)
    lo = int(np.floor(np.min(values)))
    hi = int(np.ceil(np.percentile(values, 99.5)))
    hi = max(hi, lo + 1)
    if (hi - lo) <= 60:
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    return np.linspace(lo, hi, 61)


def _pt_bins(values: np.ndarray, *, n_bins: int = 50) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    if positive.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1), "linear"
    lo = max(float(np.percentile(positive, 0.5)), float(np.min(positive)))
    hi = max(float(np.percentile(positive, 99.5)), lo * 1.05)
    return np.geomspace(lo, hi, n_bins + 1), "log"


def _particle_count_map(hits_raw: pd.DataFrame) -> pd.Series:
    return hits_raw["particle_id"].value_counts()


def _apply_truth_particle_selection(
    particles_raw: pd.DataFrame,
    hit_counts_pre: pd.Series,
    *,
    min_pt: float,
    max_abs_eta: float,
    min_num_hits: int,
    event_max_num_particles: int,
    strict_max_objects: bool,
) -> pd.DataFrame:
    truth_particles = particles_raw.copy()
    truth_particles["num_hits_pre"] = truth_particles["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)
    truth_particles = truth_particles[truth_particles["pt"] > min_pt].copy()
    truth_particles = truth_particles[np.abs(truth_particles["eta"]) < max_abs_eta].copy()
    truth_particles = truth_particles[truth_particles["num_hits_pre"] >= min_num_hits].copy()
    if len(truth_particles) > event_max_num_particles:
        if strict_max_objects:
            raise ValueError(
                f"Event exceeds event_max_num_particles: {len(truth_particles)} > {event_max_num_particles}"
            )
        truth_particles = truth_particles.iloc[:event_max_num_particles].copy()
    return truth_particles


def _build_event_views(cfg: dict, hits_raw: pd.DataFrame, particles_raw: pd.DataFrame, *, lowpt_threshold: float) -> EventViews:
    data_cfg = cfg["data"]
    hit_counts_pre = _particle_count_map(hits_raw)

    truth_particles = _apply_truth_particle_selection(
        particles_raw,
        hit_counts_pre,
        min_pt=float(data_cfg["particle_min_pt"]),
        max_abs_eta=float(data_cfg["particle_max_abs_eta"]),
        min_num_hits=int(data_cfg["particle_min_num_hits"]),
        event_max_num_particles=int(data_cfg["event_max_num_particles"]),
        strict_max_objects=bool(data_cfg.get("strict_max_objects", False)),
    )
    truth_particle_ids = truth_particles["particle_id"].to_numpy(dtype=np.int64, copy=False)
    truth_hits = hits_raw[hits_raw["particle_id"].isin(truth_particle_ids)].copy()

    lowpt_particles = particles_raw.copy()
    lowpt_particles["num_hits_pre"] = lowpt_particles["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)
    lowpt_particles = lowpt_particles[lowpt_particles["pt"] <= float(lowpt_threshold)].copy()
    lowpt_particle_ids = lowpt_particles["particle_id"].to_numpy(dtype=np.int64, copy=False)
    lowpt_hits = hits_raw[hits_raw["particle_id"].isin(lowpt_particle_ids)].copy()

    noise_hits = hits_raw[hits_raw["particle_id"] == 0].copy()

    truth_p = truth_particles["p"].to_numpy(dtype=np.float64, copy=False)
    truth_metrics = {
        "sum_truth_px": float(truth_particles["px"].sum()) if not truth_particles.empty else 0.0,
        "sum_truth_py": float(truth_particles["py"].sum()) if not truth_particles.empty else 0.0,
        "sum_truth_pz": float(truth_particles["pz"].sum()) if not truth_particles.empty else 0.0,
        "sum_truth_p": float(np.sum(truth_p)) if truth_p.size > 0 else 0.0,
        "sum_truth_pt": float(np.sum(truth_particles["pt"].to_numpy(dtype=np.float64, copy=False))) if not truth_particles.empty else 0.0,
        "sum_truth_eta": float(np.sum(truth_particles["eta"].to_numpy(dtype=np.float64, copy=False))) if not truth_particles.empty else 0.0,
        "num_truth_particles": int(len(truth_particles)),
        "num_truth_hits": int(len(truth_hits)),
        "num_raw_hits": int(len(hits_raw)),
        "num_noise_hits": int(len(noise_hits)),
        "noise_hit_fraction": float(len(noise_hits) / max(len(hits_raw), 1)),
        "num_lowpt_particles": int(len(lowpt_particles)),
        "num_lowpt_hits": int(len(lowpt_hits)),
        "lowpt_hit_fraction": float(len(lowpt_hits) / max(len(hits_raw), 1)),
    }

    return EventViews(
        hits_raw=hits_raw,
        particles_raw=particles_raw,
        truth_particles=truth_particles,
        truth_hits=truth_hits,
        lowpt_particles=lowpt_particles,
        lowpt_hits=lowpt_hits,
        noise_hits=noise_hits,
        metrics=truth_metrics,
    )


def _resolve_band_assignments(cfg: dict, split: str, args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    data_cfg = cfg["data"]
    split_dir = Path(data_cfg[f"{split}_dir"])

    if args.band_assignments_csv is not None:
        band_df = pd.read_csv(args.band_assignments_csv)
        band_df = band_df[band_df["split"] == split].reset_index(drop=True)
        if band_df.empty:
            raise ValueError(f"No rows found for split '{split}' in {args.band_assignments_csv}.")
        if args.max_events > 0:
            band_df = band_df.iloc[: args.max_events].reset_index(drop=True)
        band_df = band_df.rename(columns={"num_reconstructable_tracks": "reconstructable_tracks_post_filter"})
        needed = [
            "split",
            "event_index",
            "sample_id",
            "event_name",
            "band",
            "hits_after_filter",
            "reconstructable_tracks_post_filter",
            "band_position",
            "upper_band_posterior",
        ]
        missing = [col for col in needed if col not in band_df.columns]
        if missing:
            raise ValueError(f"Band assignments CSV is missing required columns: {missing}")
        return band_df.loc[:, needed].copy(), split_dir

    split_counts = _collect_counts(
        cfg=cfg,
        split=split,
        max_events=args.max_events,
        use_hit_eval=True,
        hit_count_mode=ALL_HITS,
    )
    assignments = _assign_event_bands(split_counts, band_clustering=args.band_clustering)
    band_df = pd.DataFrame(
        {
            "split": split,
            "event_index": assignments.event_indices.astype(np.int32),
            "sample_id": np.asarray(assignments.sample_ids, dtype=np.int32),
            "event_name": assignments.event_names,
            "band": np.asarray([BAND_NAMES[int(code)] for code in assignments.band_codes], dtype=object),
            "hits_after_filter": assignments.hits.astype(np.int32),
            "reconstructable_tracks_post_filter": assignments.particles.astype(np.int32),
            "band_position": assignments.band_position.astype(np.float64),
            "upper_band_posterior": assignments.upper_band_posterior.astype(np.float64),
        }
    )
    return band_df, split_counts.split_dir


def _sample_band_events(assignments_df: pd.DataFrame, *, num_events: int, rng: np.random.Generator) -> pd.DataFrame:
    sampled_frames: list[pd.DataFrame] = []
    for band_name in BAND_NAMES:
        band_df = assignments_df[assignments_df["band"] == band_name].reset_index(drop=True)
        if band_df.empty:
            continue
        n_take = min(num_events, len(band_df))
        chosen = rng.choice(len(band_df), size=n_take, replace=False)
        sampled = band_df.iloc[np.sort(chosen)].copy()
        sampled["sample_rank_within_band"] = np.arange(1, len(sampled) + 1, dtype=np.int32)
        sampled_frames.append(sampled)
    if not sampled_frames:
        return pd.DataFrame(columns=list(assignments_df.columns) + ["sample_rank_within_band"])
    return pd.concat(sampled_frames, ignore_index=True)


def _safe_event_slug(event_name: str) -> str:
    return event_name.replace("/", "_")


def _plot_hist(ax, values: np.ndarray, *, bins: np.ndarray, color: str, title: str, xlabel: str, density: bool, xscale: str = "linear") -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size > 0:
        ax.hist(
            values,
            bins=bins,
            density=density,
            histtype="stepfilled",
            linewidth=1.2,
            alpha=0.75,
            color=color,
        )
    else:
        ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#666666")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    ax.set_axisbelow(True)
    if xscale == "log":
        ax.set_xscale("log")


def _plot_xy_overview(ax, event: EventViews, *, band_color: str) -> None:
    plt = _get_pyplot()

    hits_raw = event.hits_raw
    truth_hits = event.truth_hits
    lowpt_hits = event.lowpt_hits
    noise_hits = event.noise_hits
    truth_particles = event.truth_particles

    ax.scatter(hits_raw["x"], hits_raw["y"], s=4, color="#cfcfcf", alpha=0.35, edgecolors="none", rasterized=True)
    if not lowpt_hits.empty:
        ax.scatter(lowpt_hits["x"], lowpt_hits["y"], s=8, color="#7570b3", alpha=0.65, edgecolors="none", rasterized=True)
    if not noise_hits.empty:
        ax.scatter(noise_hits["x"], noise_hits["y"], s=9, color="#d95f02", alpha=0.80, edgecolors="none", rasterized=True)
    if not truth_hits.empty:
        ax.scatter(truth_hits["x"], truth_hits["y"], s=10, color=band_color, alpha=0.78, edgecolors="none", rasterized=True)

    if not truth_particles.empty and not truth_hits.empty:
        plot_particles = truth_particles.sort_values(["pt", "num_hits_pre"], ascending=[False, False])
        track_colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(plot_particles), 1)))
        max_radius = max(float(hits_raw["r"].max()) if not hits_raw.empty else 0.0, 1.0)
        for color, (_, particle_row) in zip(track_colors, plot_particles.iterrows(), strict=False):
            particle_hits = truth_hits[truth_hits["particle_id"] == particle_row["particle_id"]].sort_values("r")
            if len(particle_hits) >= 2:
                ax.plot(
                    particle_hits["x"],
                    particle_hits["y"],
                    color=color,
                    linewidth=1.2,
                    alpha=0.82,
                )
            ax.plot(
                [0.0, 1.05 * max_radius * np.cos(float(particle_row["phi"]))],
                [0.0, 1.05 * max_radius * np.sin(float(particle_row["phi"]))],
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
            )

    ax.set_title("x-y hit view with truth-track overlays", fontsize=10.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    ax.set_axisbelow(True)


def _plot_sampled_event_figure(
    event: EventViews,
    event_row: pd.Series,
    *,
    output_path: Path,
    lowpt_threshold: float,
    density: bool,
) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(3, 4, figsize=(16.4, 11.6), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.28)
    axes_flat = axes.ravel()

    truth_pt = event.truth_particles["pt"].to_numpy(dtype=np.float64, copy=False)
    truth_eta = event.truth_particles["eta"].to_numpy(dtype=np.float64, copy=False)
    truth_phi = event.truth_particles["phi"].to_numpy(dtype=np.float64, copy=False)
    truth_hits_per_particle = event.truth_particles["num_hits_pre"].to_numpy(dtype=np.float64, copy=False)
    truth_hit_r = event.truth_hits["r"].to_numpy(dtype=np.float64, copy=False)
    lowpt_pt = event.lowpt_particles["pt"].to_numpy(dtype=np.float64, copy=False)
    lowpt_eta = event.lowpt_particles["eta"].to_numpy(dtype=np.float64, copy=False)
    lowpt_hits_per_particle = event.lowpt_particles["num_hits_pre"].to_numpy(dtype=np.float64, copy=False)
    noise_r = event.noise_hits["r"].to_numpy(dtype=np.float64, copy=False)
    noise_eta = event.noise_hits["eta"].to_numpy(dtype=np.float64, copy=False)
    noise_phi = event.noise_hits["phi"].to_numpy(dtype=np.float64, copy=False)

    band_name = str(event_row["band"])
    band_color = BAND_COLORS[band_name]

    bins, xscale = _pt_bins(truth_pt)
    _plot_hist(axes_flat[0], truth_pt, bins=bins, color=band_color, title="Truth particle pT", xlabel="pT [GeV]", density=density, xscale=xscale)
    _plot_hist(axes_flat[1], truth_eta, bins=_continuous_bins(truth_eta), color=band_color, title="Truth particle eta", xlabel="eta", density=density)
    _plot_hist(axes_flat[2], truth_hits_per_particle, bins=_count_bins(truth_hits_per_particle), color=band_color, title="Truth particle hit count", xlabel="truth hits / particle", density=density)
    _plot_hist(axes_flat[3], truth_hit_r, bins=_continuous_bins(truth_hit_r), color=band_color, title="Truth-hit radius", xlabel="r [m]", density=density)

    bins, xscale = _pt_bins(lowpt_pt)
    _plot_hist(axes_flat[4], lowpt_pt, bins=bins, color="#7570b3", title=f"pT <= {lowpt_threshold:.2f} GeV particle pT", xlabel="pT [GeV]", density=density, xscale=xscale)
    _plot_hist(axes_flat[5], lowpt_eta, bins=_continuous_bins(lowpt_eta), color="#7570b3", title=f"pT <= {lowpt_threshold:.2f} GeV particle eta", xlabel="eta", density=density)
    _plot_hist(axes_flat[6], lowpt_hits_per_particle, bins=_count_bins(lowpt_hits_per_particle), color="#7570b3", title=f"pT <= {lowpt_threshold:.2f} GeV particle hit count", xlabel="truth hits / particle", density=density)
    _plot_hist(axes_flat[7], truth_phi, bins=np.linspace(-np.pi, np.pi, 50), color=band_color, title="Truth particle phi", xlabel="phi [rad]", density=density)

    _plot_hist(axes_flat[8], noise_r, bins=_continuous_bins(noise_r), color="#d95f02", title="Noise-hit radius", xlabel="r [m]", density=density)
    _plot_hist(axes_flat[9], noise_eta, bins=_continuous_bins(noise_eta), color="#d95f02", title="Noise-hit eta", xlabel="eta", density=density)
    _plot_hist(axes_flat[10], noise_phi, bins=np.linspace(-np.pi, np.pi, 50), color="#d95f02", title="Noise-hit phi", xlabel="phi [rad]", density=density)
    _plot_xy_overview(axes_flat[11], event, band_color=band_color)

    summary_lines = [
        f"band = {band_name.replace('_', ' ')}",
        f"band position = {float(event_row['band_position']):.3f}",
        f"filtered hits = {int(event_row['hits_after_filter'])}",
        f"post-filter reco tracks = {int(event_row['reconstructable_tracks_post_filter'])}",
        f"truth particles = {int(event.metrics['num_truth_particles'])}",
        f"truth hits = {int(event.metrics['num_truth_hits'])}",
        f"noise hits = {int(event.metrics['num_noise_hits'])}",
        f"low-pT particles = {int(event.metrics['num_lowpt_particles'])}",
        f"low-pT hits = {int(event.metrics['num_lowpt_hits'])}",
    ]
    axes_flat[11].text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=axes_flat[11].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle(
        f"{event_row['split']} | {event_row['event_name']} | sample #{int(event_row['sample_rank_within_band'])} in {band_name.replace('_', ' ')}",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_band_xy_sheet(
    sampled_events: list[tuple[pd.Series, EventViews]],
    *,
    output_path: Path,
    band_name: str,
) -> None:
    if not sampled_events:
        return

    plt = _get_pyplot()
    n_events = len(sampled_events)
    n_cols = min(5, max(1, n_events))
    n_rows = int(np.ceil(n_events / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.6 * n_rows), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.28, wspace=0.22)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (event_row, event) in zip(axes_flat, sampled_events, strict=False):
        _plot_xy_overview(
            ax,
            event,
            band_color=BAND_COLORS[band_name],
        )
        ax.set_title(
            f"#{int(event_row['sample_rank_within_band'])} {event_row['event_name']}",
            fontsize=9.5,
        )
        ax.text(
            0.02,
            0.98,
            (
                f"idx={int(event_row['event_index'])}\n"
                f"sid={int(event_row['sample_id'])}\n"
                f"pos={float(event_row['band_position']):.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.80, "boxstyle": "round,pad=0.18"},
        )

    for ax in axes_flat[n_events:]:
        ax.axis("off")

    fig.suptitle(
        f"Sampled x-y event views: {band_name.replace('_', ' ')}",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_event_level_summary(metrics_df: pd.DataFrame, output_path: Path, *, density: bool, lowpt_threshold: float) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(4, 4, figsize=(17.0, 13.8), constrained_layout=False)
    fig.subplots_adjust(top=0.93, hspace=0.36, wspace=0.30)
    axes_flat = axes.ravel()

    metric_specs = [
        ("sum_truth_px", "Sum truth px [GeV]"),
        ("sum_truth_py", "Sum truth py [GeV]"),
        ("sum_truth_pz", "Sum truth pz [GeV]"),
        ("sum_truth_p", "Scalar sum |p| [GeV]"),
        ("sum_truth_pt", "Scalar sum pT [GeV]"),
        ("sum_truth_eta", "Sum truth eta"),
        ("num_truth_particles", "Reconstructable truth particles / event"),
        ("num_truth_hits", "Truth hits / event"),
        ("num_raw_hits", "Raw hits / event"),
        ("hits_after_filter", "Hits after filtering / event"),
        ("reconstructable_tracks_post_filter", "Reco truth tracks after filter / event"),
        ("num_noise_hits", "Noise hits / event"),
        ("noise_hit_fraction", "Noise-hit fraction"),
        ("num_lowpt_particles", f"Particles with pT <= {lowpt_threshold:.2f} GeV / event"),
        ("num_lowpt_hits", f"Hits from pT <= {lowpt_threshold:.2f} GeV particles / event"),
        ("lowpt_hit_fraction", "Low-pT-hit fraction"),
    ]

    for ax, (column, title) in zip(axes_flat, metric_specs, strict=True):
        lower_vals = metrics_df.loc[metrics_df["band"] == BAND_NAMES[0], column].to_numpy(dtype=np.float64)
        upper_vals = metrics_df.loc[metrics_df["band"] == BAND_NAMES[1], column].to_numpy(dtype=np.float64)
        combined = np.concatenate([lower_vals[np.isfinite(lower_vals)], upper_vals[np.isfinite(upper_vals)]]) if (len(lower_vals) + len(upper_vals)) > 0 else np.array([], dtype=np.float64)
        if "fraction" in column:
            bins = np.linspace(0.0, 1.0, 41)
        elif column.startswith("num_") or column in {"hits_after_filter", "reconstructable_tracks_post_filter"}:
            bins = _count_bins(combined)
        else:
            bins = _continuous_bins(combined)
        ax.hist(lower_vals[np.isfinite(lower_vals)], bins=bins, density=density, histtype="step", linewidth=1.5, color=BAND_COLORS[BAND_NAMES[0]], label="lower band")
        ax.hist(upper_vals[np.isfinite(upper_vals)], bins=bins, density=density, histtype="step", linewidth=1.5, color=BAND_COLORS[BAND_NAMES[1]], label="upper band")
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("Density" if density else "Events")
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    axes_flat[0].legend(frameon=False, loc="best")
    fig.suptitle("All-event summaries by post-filter band", fontsize=13, y=0.975)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_event_ids_by_band(assignments_df: pd.DataFrame, output_path: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(top=0.87, wspace=0.28)

    y_positions = {BAND_NAMES[0]: 0.0, BAND_NAMES[1]: 1.0}
    y_labels = {BAND_NAMES[0]: "lower band", BAND_NAMES[1]: "upper band"}

    for ax, x_col, xlabel in (
        (axes[0], "event_index", "Event index"),
        (axes[1], "sample_id", "Sample id"),
    ):
        for band_name in BAND_NAMES:
            band_df = assignments_df[assignments_df["band"] == band_name].copy()
            if band_df.empty:
                continue
            offsets = np.linspace(-0.08, 0.08, len(band_df)) if len(band_df) > 1 else np.array([0.0], dtype=np.float64)
            ax.scatter(
                band_df[x_col].to_numpy(dtype=np.float64, copy=False),
                y_positions[band_name] + offsets,
                s=16,
                alpha=0.75,
                color=BAND_COLORS[band_name],
                edgecolors="none",
                rasterized=True,
                label=y_labels[band_name],
            )
        ax.set_xlabel(xlabel)
        ax.set_yticks([y_positions[BAND_NAMES[0]], y_positions[BAND_NAMES[1]]], [y_labels[BAND_NAMES[0]], y_labels[BAND_NAMES[1]]])
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    axes[0].set_title("Event index membership by band")
    axes[1].set_title("Sample id membership by band")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("All-event band membership overview", fontsize=13, y=0.965)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = _read_yaml(args.config)
    data_cfg = cfg["data"]
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_rng = np.random.default_rng(args.sample_seed)
        assignments_df, split_dir = _resolve_band_assignments(cfg, split, args)
        sampled_df = _sample_band_events(assignments_df, num_events=args.sample_events_per_band, rng=split_rng)
        sampled_key_set = set(
            zip(
                sampled_df["event_name"].astype(str),
                sampled_df["band"].astype(str),
            )
        )

        split_output_dir = args.output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(split_output_dir / f"{args.output_stem}_{split}_sampled_events.csv", index=False)

        print(
            f"[{split}] processing {len(assignments_df)} events total; "
            f"sampled {sum(sampled_df['band'] == BAND_NAMES[0])} lower-band and "
            f"{sum(sampled_df['band'] == BAND_NAMES[1])} upper-band events"
        )

        all_event_rows: list[dict[str, float | int | str]] = []
        sampled_xy_events: dict[str, list[tuple[pd.Series, EventViews]]] = {band_name: [] for band_name in BAND_NAMES}
        for idx, event_row in assignments_df.iterrows():
            hits_raw, particles_raw = _load_raw_trackml_event(split_dir, str(event_row["event_name"]), hit_volume_ids)
            event = _build_event_views(cfg, hits_raw, particles_raw, lowpt_threshold=args.lowpt_threshold)

            metrics_row = {
                **{col: event_row[col] for col in assignments_df.columns},
                **event.metrics,
            }
            all_event_rows.append(metrics_row)

            sample_key = (str(event_row["event_name"]), str(event_row["band"]))
            if sample_key in sampled_key_set:
                sampled_row = sampled_df[
                    (sampled_df["event_name"] == event_row["event_name"])
                    & (sampled_df["band"] == event_row["band"])
                ].iloc[0]
                event_band_dir = split_output_dir / str(event_row["band"])
                event_band_dir.mkdir(parents=True, exist_ok=True)
                out_base = event_band_dir / (
                    f"{args.output_stem}_{split}_{event_row['band']}_{int(sampled_row['sample_rank_within_band']):02d}_{_safe_event_slug(str(event_row['event_name']))}"
                )
                _plot_sampled_event_figure(
                    event,
                    sampled_row,
                    output_path=out_base,
                    lowpt_threshold=args.lowpt_threshold,
                    density=args.hist_density,
                )
                sampled_xy_events[str(event_row["band"])].append((sampled_row, event))

            if (idx + 1) % 50 == 0 or (idx + 1) == len(assignments_df):
                print(f"[{split}] processed {idx + 1}/{len(assignments_df)} events")

        metrics_df = pd.DataFrame(all_event_rows)
        metrics_csv = split_output_dir / f"{args.output_stem}_{split}_event_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        _plot_event_level_summary(
            metrics_df,
            split_output_dir / f"{args.output_stem}_{split}_all_event_summary",
            density=args.hist_density,
            lowpt_threshold=args.lowpt_threshold,
        )
        _plot_event_ids_by_band(
            assignments_df,
            split_output_dir / f"{args.output_stem}_{split}_event_ids_by_band",
        )
        for band_name in BAND_NAMES:
            _plot_band_xy_sheet(
                sampled_xy_events[band_name],
                output_path=split_output_dir / f"{args.output_stem}_{split}_{band_name}_xy_sheet",
                band_name=band_name,
            )
        print(f"[{split}] wrote sampled event figures under {split_output_dir}")
        print(f"[{split}] wrote event metric table to {metrics_csv}")


if __name__ == "__main__":
    main()
