#!/usr/bin/env python3
"""Paper-style plots of reconstructable tracks versus event hit counts."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900.yaml")
DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots/reconstructable_vs_post_filter_hits_train_070526/")
VALID_SPLITS = ("train", "val", "test")
VALID_HIT_SELECTIONS = ("all", "valid-track")
VALID_BAND_CLUSTERING = ("parallel-lines", "kmeans", "median")
VALID_HIST_MODES = ("density", "count", "count-per-event")
BAND_NAMES = ("lower_band", "upper_band")
POSITION_GROUPS = ("lower_core", "between_bands", "upper_core")
BAND_COLORS = {
    "lower_band": "#d95f02",
    "upper_band": "#1b9e77",
}
HIT_FEATURE_COLUMNS = ("x", "y", "z", "r", "eta", "phi")
PARTICLE_FEATURE_COLUMNS = ("pt", "eta", "phi", "num_hits_post", "num_hits_pre", "hit_retention")
TRUE_TRACK_FEATURE_COLUMNS = ("pt", "eta", "phi", "num_true_hits")
EXPLAINER_EXCLUDED_COLUMNS = {
    "split",
    "event_index",
    "sample_id",
    "event_name",
    "band",
    "band_position",
    "upper_band_posterior",
    "position_group",
    "hits_after_filter",
    "reconstructable_tracks_post_filter",
    "lost_reconstructable_tracks",
    "reconstructable_track_retention",
}
PLOT_HIST_MODE = "density"
POWERPOINT_BAND_PLOT_SECTIONS = frozenset(
    {
        "assignments",
        "scatter",
        "feature_comparison",
        "truth_feature_comparison",
        "threshold_sweep",
        "invalid_source_comparison",
        "noise_hit_comparison",
        "nonreco_failure_breakdown",
        "nonreco_reason_conditioned",
        "prefilter_lowpt_noise_comparison",
        "prefilter_truth_nonreco_comparison",
        "prefilter_truth_nonreco_event_metrics",
        "prefilter_truth_noise_comparison",
        "prefilter_lowpt_particle_kinematics",
    }
)


@dataclass
class SplitCounts:
    split: str
    event_names: list[str]
    sample_ids: list[int]
    hits: np.ndarray
    particles: np.ndarray
    split_dir: Path
    hit_eval_path: Path | None
    pre_filter_hits: np.ndarray | None = None


ALL_HITS = "all_hits"
VALID_TRACK_HITS = "valid_track_hits"


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float


@dataclass
class BandAssignments:
    split: str
    event_indices: np.ndarray
    event_names: list[str]
    sample_ids: list[int]
    hits: np.ndarray
    particles: np.ndarray
    fitted_particles: np.ndarray
    residuals: np.ndarray
    band_codes: np.ndarray
    band_centers: np.ndarray
    model_slope: float
    band_intercepts: np.ndarray
    band_position: np.ndarray
    upper_band_posterior: np.ndarray
    split_dir: Path
    hit_eval_path: Path | None


@dataclass
class BandFeatureSamples:
    split: str
    hit_selection: str
    hit_samples: dict[str, pd.DataFrame]
    particle_samples: dict[str, pd.DataFrame]
    event_hits_per_track: dict[str, np.ndarray]
    totals: dict[str, dict[str, int]]


@dataclass
class BandExplainerResults:
    split: str
    feature_columns: list[str]
    residualized_features: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame
    summary: dict[str, float]


@dataclass
class TruthBandSamples:
    split: str
    hit_samples: dict[str, pd.DataFrame]
    track_samples: dict[str, pd.DataFrame]
    event_metrics: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, int]]


@dataclass
class BandTrackLossSamples:
    split: str
    track_samples: dict[str, pd.DataFrame]
    event_metrics: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, int]]


@dataclass
class BandHitFilterDiagnostics:
    split: str
    score_samples: pd.DataFrame
    threshold_sweep: pd.DataFrame
    region_summary: pd.DataFrame
    invalid_category_summary: pd.DataFrame


@dataclass
class BandInvalidSourceSamples:
    split: str
    nonreco_hit_samples: dict[str, pd.DataFrame]
    nonreco_particle_samples: dict[str, pd.DataFrame]
    noise_hit_samples: dict[str, pd.DataFrame]
    event_metrics: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, int]]
    failure_summary: pd.DataFrame


@dataclass
class BandPrefilterLowPtNoiseSamples:
    split: str
    lowpt_hit_samples: dict[str, pd.DataFrame]
    lowpt_particle_samples: dict[str, pd.DataFrame]
    noise_hit_samples: dict[str, pd.DataFrame]
    event_metrics: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, int]]
    event_diagnostics: pd.DataFrame
    region_summary: pd.DataFrame


@dataclass
class BandPrefilterTruthNonrecoSamples:
    split: str
    particle_samples: dict[str, pd.DataFrame]
    event_metrics: dict[str, dict[str, np.ndarray]]
    totals: dict[str, dict[str, int]]
    noise_hit_samples: dict[str, pd.DataFrame]
    noise_event_metrics: dict[str, dict[str, np.ndarray]]


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
        "--hist-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy shortcut for density-vs-count histograms when --hist-mode is not set.",
    )
    parser.add_argument(
        "--hist-mode",
        choices=VALID_HIST_MODES,
        default=None,
        help="Histogram normalization mode. When set, this overrides --hist-density.",
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
    parser.add_argument(
        "--load-from-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing saved outputs from a previous run. "
            "When provided, the script reuses saved per-event tables and any available "
            "band-analysis caches before falling back to recomputation."
        ),
    )
    parser.add_argument(
        "--save-band-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist reusable band-analysis sample caches (*.pkl) alongside the usual CSV "
            "summaries so later reruns can replot without re-reading raw events."
        ),
    )
    parser.add_argument(
        "--analyze-bands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split post-filter events into two residual bands and compare hit/particle features across them.",
    )
    parser.add_argument(
        "--powerpoint-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run only the plot families used in Band-analysis-pt900.pptx. "
            "This enables --analyze-bands and skips unrelated band-analysis sections."
        ),
    )
    parser.add_argument(
        "--band-hit-selection",
        choices=VALID_HIT_SELECTIONS,
        default="valid-track",
        help="Which filtered hits to use for hit-level band comparisons.",
    )
    parser.add_argument(
        "--band-clustering",
        choices=VALID_BAND_CLUSTERING,
        default="parallel-lines",
        help="How to split events into two post-filter bands.",
    )
    parser.add_argument(
        "--band-max-hit-samples-per-band",
        type=int,
        default=200_000,
        help="Maximum number of hit rows to retain per band for hit-level comparison plots.",
    )
    parser.add_argument(
        "--band-max-particle-samples-per-band",
        type=int,
        default=100_000,
        help="Maximum number of particle rows to retain per band for particle-level comparison plots.",
    )
    parser.add_argument(
        "--band-max-score-samples-per-band",
        type=int,
        default=200_000,
        help="Maximum number of pre-threshold hit-score rows to retain per band for score diagnostics.",
    )
    parser.add_argument(
        "--band-rng-seed",
        type=int,
        default=12345,
        help="Random seed used for bounded per-band feature sampling.",
    )
    parser.add_argument(
        "--band-gap-half-width",
        type=float,
        default=0.15,
        help=(
            "Half-width of the between-band region in normalized band-position coordinates. "
            "A value of 0.15 means lower/upper cores are outside [0.35, 0.65]."
        ),
    )
    parser.add_argument(
        "--band-explainer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit a simple event-level band explainer using residualized diagnostics features.",
    )
    parser.add_argument(
        "--band-explainer-hit-bins",
        type=int,
        default=20,
        help="Number of hit-count quantile bins used to residualize event features before band explanation.",
    )
    parser.add_argument(
        "--band-explainer-top-features",
        type=int,
        default=15,
        help="Number of top features to highlight in the band explainer plot.",
    )
    parser.add_argument(
        "--band-explainer-test-fraction",
        type=float,
        default=0.2,
        help="Fraction of lower/upper-core events held out for band-explainer evaluation.",
    )
    parser.add_argument(
        "--band-threshold-sweep-points",
        type=int,
        default=41,
        help="Number of score thresholds to evaluate between 0 and 1 for the band threshold sweep.",
    )
    parser.add_argument(
        "--band-crowding-distance",
        type=float,
        default=0.05,
        help="Same-layer r*delta_phi/z distance threshold [m] used for close-hit ambiguity diagnostics.",
    )
    parser.add_argument(
        "--band-top-region-keys",
        type=int,
        default=12,
        help="Number of top detector volume/layer regions to highlight in region-conditioned diagnostics.",
    )
    parser.add_argument(
        "--skip-if-not-cached",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When set, any sample type whose cache is not found in --load-from-output-dir is skipped "
            "entirely (no recomputation, no related plots). Useful for rapidly regenerating plots for "
            "only the sample types that are already cached."
        ),
    )
    return parser.parse_args()


def _hist_density() -> bool:
    return PLOT_HIST_MODE == "density"


def _hist_count_per_event() -> bool:
    return PLOT_HIST_MODE == "count-per-event"


def _hist_ylabel(*, base_count_label: str = "Count", allow_count_per_event: bool = False) -> str:
    if _hist_density():
        return "Density"
    if allow_count_per_event and _hist_count_per_event():
        return f"{base_count_label} / event"
    return base_count_label


def _event_hist_ylabel(*, per_band: bool = False) -> str:
    if _hist_density():
        return "Density"
    if _hist_count_per_event():
        return "Fraction of band events" if per_band else "Fraction of events"
    return "Events"


def _hist_weights(num_values: int, *, num_events: int, allow_count_per_event: bool = False) -> np.ndarray | None:
    if not allow_count_per_event or not _hist_count_per_event():
        return None
    return np.full(num_values, 1.0 / max(int(num_events), 1), dtype=np.float64)


def _band_hist_weights(num_values: int, *, num_events: int) -> np.ndarray | None:
    return _hist_weights(num_values, num_events=num_events, allow_count_per_event=True)


def _sampled_population_hist_weights(
    num_values: int,
    *,
    sampled_population_size: int,
    total_population: int,
    num_events: int,
) -> np.ndarray | None:
    if _hist_density():
        return None
    if num_values <= 0 or sampled_population_size <= 0:
        return None

    scale = float(total_population) / float(max(int(sampled_population_size), 1))
    if _hist_count_per_event():
        scale /= float(max(int(num_events), 1))
    return np.full(num_values, scale, dtype=np.float64)


def _band_mixed_hist_ylabel(feature_key: str) -> str:
    if feature_key.startswith("event_"):
        return _event_hist_ylabel(per_band=True)
    return _hist_ylabel(allow_count_per_event=True)


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


def _load_raw_trackml_event(split_dir: Path, event_name: str, hit_volume_ids: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    particles = pd.read_parquet(split_dir / Path(event_name + "-parts.parquet"))
    hits = pd.read_parquet(split_dir / Path(event_name + "-hits.parquet"))

    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)].copy()
    else:
        hits = hits.copy()
    particles = particles.copy()

    for coord in ["x", "y", "z"]:
        hits[coord] *= 0.01

    hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
    hits["theta"] = np.arccos(hits["z"] / hits["s"])
    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
    hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

    particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    particles["qopt"] = particles["q"] / particles["pt"]
    particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
    particles["theta"] = np.arccos(particles["pz"] / particles["p"])
    particles["phi"] = np.arctan2(particles["py"], particles["px"])
    particles["costheta"] = np.cos(particles["theta"])
    particles["sintheta"] = np.sin(particles["theta"])
    particles["cosphi"] = np.cos(particles["phi"])
    particles["sinphi"] = np.sin(particles["phi"])
    return hits, particles


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
    event_names = dataset.event_names[:n_use]
    sample_ids = dataset.sample_ids[:n_use]
    # When loading raw (unfiltered) events, record all-hit counts for free as the
    # pre-filter reference even when the primary hit_count_mode is something else.
    pre_filter = np.zeros(n_use, dtype=np.int32) if not use_hit_eval else None

    print(f"[{split}] processing {n_use} events from {split_dir} with hit-eval {hit_eval_path} ({hit_count_mode})")
    for i in range(n_use):
        event_hits, event_particles = dataset.load_event(i)
        if hit_count_mode == ALL_HITS:
            hits[i] = len(event_hits)
        elif hit_count_mode == VALID_TRACK_HITS:
            if "on_valid_particle" not in event_hits.columns:
                raise KeyError("Expected 'on_valid_particle' column on hits DataFrame.")
            hits[i] = int(event_hits["on_valid_particle"].sum())
        else:
            raise ValueError(f"Unknown hit_count_mode={hit_count_mode!r}")
        if pre_filter is not None:
            pre_filter[i] = len(event_hits)
        particles[i] = len(event_particles)
        if (i + 1) % 50 == 0 or (i + 1) == n_use:
            print(f"[{split}] processed {i + 1}/{n_use} events")

    return SplitCounts(
        split=split,
        event_names=event_names,
        sample_ids=sample_ids,
        hits=hits,
        particles=particles,
        split_dir=split_dir,
        hit_eval_path=hit_eval_path,
        pre_filter_hits=pre_filter,
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


def _median_band_split(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = float(np.median(residuals))
    band_codes = (residuals >= median).astype(np.int32)
    if np.all(band_codes == band_codes[0]):
        order = np.argsort(residuals)
        half = max(1, residuals.size // 2)
        band_codes = np.ones(residuals.size, dtype=np.int32)
        band_codes[order[:half]] = 0
    centers = np.array(
        [
            float(residuals[band_codes == 0].mean()),
            float(residuals[band_codes == 1].mean()),
        ],
        dtype=np.float64,
    )
    return band_codes, centers


def _kmeans_1d_band_split(residuals: np.ndarray, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if residuals.size < 2 or np.allclose(residuals, residuals[0]):
        return _median_band_split(residuals)

    centers = np.quantile(residuals, [0.25, 0.75]).astype(np.float64)
    if np.isclose(centers[0], centers[1]):
        return _median_band_split(residuals)

    for _ in range(max_iters):
        distances = np.abs(residuals[:, None] - centers[None, :])
        band_codes = np.argmin(distances, axis=1).astype(np.int32)
        if np.unique(band_codes).size < 2:
            return _median_band_split(residuals)

        new_centers = centers.copy()
        for code in range(2):
            new_centers[code] = float(residuals[band_codes == code].mean())

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    order = np.argsort(centers)
    remap = np.zeros(2, dtype=np.int32)
    remap[order[0]] = 0
    remap[order[1]] = 1
    band_codes = remap[band_codes]
    centers = np.sort(centers)
    return band_codes, centers


def _parallel_line_mixture_split(
    hits: np.ndarray,
    particles: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    x = hits.astype(np.float64)
    y = particles.astype(np.float64)

    base_fit = _linear_fit(hits, particles)
    if base_fit is None:
        raise RuntimeError("Need at least two distinct hit counts to fit parallel bands.")

    beta = float(base_fit.slope)
    baseline = float(base_fit.intercept)
    residuals = y - (baseline + beta * x)
    init_codes, init_centers = _kmeans_1d_band_split(residuals)

    alpha = np.array(
        [
            baseline + float(init_centers[0]),
            baseline + float(init_centers[1]),
        ],
        dtype=np.float64,
    )
    pi = np.array(
        [
            max(float(np.mean(init_codes == 0)), 1e-3),
            max(float(np.mean(init_codes == 1)), 1e-3),
        ],
        dtype=np.float64,
    )
    pi /= pi.sum()
    sigma2 = np.array(
        [
            max(float(np.var(y[init_codes == 0] - (alpha[0] + beta * x[init_codes == 0]))), 1.0),
            max(float(np.var(y[init_codes == 1] - (alpha[1] + beta * x[init_codes == 1]))), 1.0),
        ],
        dtype=np.float64,
    )
    sigma2 = np.maximum(sigma2, 1.0)

    n = len(x)
    prev_loglike = -np.inf
    responsibilities = np.full((n, 2), 0.5, dtype=np.float64)

    for _ in range(max_iters):
        means = np.stack([alpha[0] + beta * x, alpha[1] + beta * x], axis=1)
        log_prob = np.empty((n, 2), dtype=np.float64)
        for k in range(2):
            var_k = max(float(sigma2[k]), 1e-6)
            log_prob[:, k] = (
                np.log(max(float(pi[k]), 1e-12))
                - 0.5 * np.log(2.0 * np.pi * var_k)
                - 0.5 * np.square(y - means[:, k]) / var_k
            )

        max_log_prob = np.max(log_prob, axis=1, keepdims=True)
        stabilized = np.exp(log_prob - max_log_prob)
        responsibilities = stabilized / stabilized.sum(axis=1, keepdims=True)

        loglike = float(np.sum(max_log_prob[:, 0] + np.log(stabilized.sum(axis=1))))
        if np.isfinite(prev_loglike) and abs(loglike - prev_loglike) < tol:
            break
        prev_loglike = loglike

        weights = responsibilities.sum(axis=0)
        weights = np.maximum(weights, 1e-6)
        pi = weights / weights.sum()

        design = np.vstack(
            [
                np.column_stack([np.ones(n), np.zeros(n), x]),
                np.column_stack([np.zeros(n), np.ones(n), x]),
            ]
        )
        response = np.concatenate([y, y])
        row_weights = np.concatenate([responsibilities[:, 0], responsibilities[:, 1]])
        sqrt_w = np.sqrt(np.maximum(row_weights, 1e-12))
        weighted_design = design * sqrt_w[:, None]
        weighted_response = response * sqrt_w
        coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_response, rcond=None)
        alpha = coeffs[:2].astype(np.float64)
        beta = float(coeffs[2])

        means = np.stack([alpha[0] + beta * x, alpha[1] + beta * x], axis=1)
        residual_matrix = y[:, None] - means
        sigma2 = np.sum(responsibilities * np.square(residual_matrix), axis=0) / weights
        sigma2 = np.maximum(sigma2, 1.0)

    order = np.argsort(alpha)
    remap = np.zeros(2, dtype=np.int32)
    remap[order[0]] = 0
    remap[order[1]] = 1
    band_codes = remap[np.argmax(responsibilities, axis=1)]
    alpha = alpha[order]
    sigma2 = sigma2[order]

    fitted_particles = alpha[band_codes] + beta * x
    residuals = y - fitted_particles
    band_centers = np.array(
        [
            float(residuals[band_codes == 0].mean()) if np.any(band_codes == 0) else 0.0,
            float(residuals[band_codes == 1].mean()) if np.any(band_codes == 1) else 0.0,
        ],
        dtype=np.float64,
    )
    upper_band_posterior = responsibilities[:, order[1]].astype(np.float64)
    return band_codes.astype(np.int32), band_centers, float(beta), alpha.astype(np.float64), upper_band_posterior


def _assign_event_bands(split_counts: SplitCounts, band_clustering: str) -> BandAssignments:
    fit = _linear_fit(split_counts.hits, split_counts.particles)
    if fit is None:
        raise RuntimeError(
            f"Cannot assign bands for split '{split_counts.split}': need at least two distinct hit counts for a linear detrend."
        )

    hits = split_counts.hits.astype(np.float64)
    particles = split_counts.particles.astype(np.float64)
    upper_band_posterior = np.full(len(hits), np.nan, dtype=np.float64)

    if band_clustering == "parallel-lines":
        band_codes, band_centers, model_slope, band_intercepts, upper_band_posterior = _parallel_line_mixture_split(
            split_counts.hits,
            split_counts.particles,
        )
        fitted_particles = band_intercepts[band_codes] + model_slope * hits
        residuals = particles - fitted_particles
        intercept_event = particles - model_slope * hits
        intercept_span = float(band_intercepts[1] - band_intercepts[0])
        if abs(intercept_span) < 1e-9:
            band_position = np.full(len(hits), 0.5, dtype=np.float64)
        else:
            band_position = (intercept_event - band_intercepts[0]) / intercept_span
    else:
        fitted_particles = fit.intercept + fit.slope * hits
        residuals = particles - fitted_particles
        model_slope = float(fit.slope)
        band_intercepts = np.array(
            [
                float(fit.intercept),
                float(fit.intercept),
            ],
            dtype=np.float64,
        )
        band_position = np.full(len(hits), 0.5, dtype=np.float64)

    if band_clustering == "kmeans":
        band_codes, band_centers = _kmeans_1d_band_split(residuals)
        center_span = float(band_centers[1] - band_centers[0]) if abs(float(band_centers[1] - band_centers[0])) > 1e-9 else 1.0
        band_position = (residuals - band_centers[0]) / center_span
        dist0 = np.abs(residuals - band_centers[0])
        dist1 = np.abs(residuals - band_centers[1])
        upper_band_posterior = np.exp(-dist1) / (np.exp(-dist0) + np.exp(-dist1))
    elif band_clustering == "median":
        band_codes, band_centers = _median_band_split(residuals)
        center_span = float(band_centers[1] - band_centers[0]) if abs(float(band_centers[1] - band_centers[0])) > 1e-9 else 1.0
        band_position = (residuals - band_centers[0]) / center_span
        dist0 = np.abs(residuals - band_centers[0])
        dist1 = np.abs(residuals - band_centers[1])
        upper_band_posterior = np.exp(-dist1) / (np.exp(-dist0) + np.exp(-dist1))
    elif band_clustering != "parallel-lines":
        raise ValueError(f"Unknown band_clustering={band_clustering!r}")

    return BandAssignments(
        split=split_counts.split,
        event_indices=np.arange(len(split_counts.hits), dtype=np.int32),
        event_names=split_counts.event_names,
        sample_ids=split_counts.sample_ids,
        hits=split_counts.hits.astype(np.int32),
        particles=split_counts.particles.astype(np.int32),
        fitted_particles=fitted_particles.astype(np.float64),
        residuals=residuals.astype(np.float64),
        band_codes=band_codes.astype(np.int32),
        band_centers=band_centers.astype(np.float64),
        model_slope=float(model_slope),
        band_intercepts=band_intercepts.astype(np.float64),
        band_position=band_position.astype(np.float64),
        upper_band_posterior=upper_band_posterior.astype(np.float64),
        split_dir=split_counts.split_dir,
        hit_eval_path=split_counts.hit_eval_path,
    )


def _event_position_group(position: float, gap_half_width: float) -> str:
    if position <= 0.5 - gap_half_width:
        return "lower_core"
    if position >= 0.5 + gap_half_width:
        return "upper_core"
    return "between_bands"


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values.astype(np.float64)))


def _count_equal(values: np.ndarray, target: int) -> int:
    return int(np.sum(values == target))


def _volume_fraction_columns(volume_ids: list[int], prefix: str) -> list[str]:
    return [f"{prefix}_volume_{volume_id}" for volume_id in volume_ids]


def _volume_layer_fraction_columns(volume_layer_keys: list[tuple[int, int]], prefix: str) -> list[str]:
    return [f"{prefix}_volume_{volume_id}_layer_{layer_id}" for volume_id, layer_id in volume_layer_keys]


def _has_h5_dataset(group, path: str) -> bool:
    try:
        _ = group[path]
    except KeyError:
        return False
    return True


def _read_hit_filter_scores(hit_eval_file, sample_id: int) -> np.ndarray | None:
    candidate_paths = (
        f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle_prob",
        f"{sample_id}/preds/final/hit_filter/key_on_valid_particle_prob",
        f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle",
    )
    for path in candidate_paths:
        if _has_h5_dataset(hit_eval_file, path):
            scores = np.asarray(hit_eval_file[path], dtype=np.float64)
            if scores.ndim >= 1 and scores.shape[0] == 1:
                scores = scores[0]
            return np.asarray(scores, dtype=np.float64)
    return None


def _percentile_or_nan(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values.astype(np.float64), q))


def _cyclic_nearest_neighbor_gaps(values: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.full(values.shape, np.nan, dtype=np.float64)

    normalized = np.mod(values + 0.5 * period, period)
    order = np.argsort(normalized)
    sorted_values = normalized[order]
    forward_gaps = np.diff(np.concatenate([sorted_values, [sorted_values[0] + period]]))
    backward_gaps = np.roll(forward_gaps, 1)
    nearest_gaps = np.minimum(forward_gaps, backward_gaps)

    gaps = np.empty(values.size, dtype=np.float64)
    gaps[order] = nearest_gaps
    return gaps


def _grouped_phi_nearest_neighbor_gaps(frame: pd.DataFrame, group_cols: tuple[str, ...]) -> np.ndarray:
    if frame.empty or "phi" not in frame.columns:
        return np.array([], dtype=np.float64)

    gap_chunks: list[np.ndarray] = []
    for _, group in frame.groupby(list(group_cols), sort=False):
        gaps = _cyclic_nearest_neighbor_gaps(group["phi"].to_numpy(dtype=np.float64, copy=False), period=2.0 * np.pi)
        finite_gaps = gaps[np.isfinite(gaps)]
        if finite_gaps.size > 0:
            gap_chunks.append(finite_gaps)

    if not gap_chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(gap_chunks)


def _wrapped_delta_phi(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    return (phi_a - phi_b + np.pi) % (2.0 * np.pi) - np.pi


def _delta_r_nearest_neighbor(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    n = eta.size
    if n <= 1:
        return np.full(n, np.nan, dtype=np.float64)

    distances = np.full((n, n), np.inf, dtype=np.float64)
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        d_eta = eta[start:stop, None] - eta[None, :]
        d_phi = _wrapped_delta_phi(phi[start:stop, None], phi[None, :])
        distances[start:stop, :] = np.sqrt(np.square(d_eta) + np.square(d_phi))
    np.fill_diagonal(distances, np.inf)
    return np.min(distances, axis=1)


def _same_layer_min_rphiz_distances(
    source_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    *,
    exclude_self: bool = False,
    group_cols: tuple[str, ...] = ("volume_id", "layer_id"),
    chunk_size: int = 256,
) -> np.ndarray:
    if source_frame.empty:
        return np.array([], dtype=np.float64)
    if reference_frame.empty:
        return np.full(len(source_frame), np.nan, dtype=np.float64)

    out = np.full(len(source_frame), np.nan, dtype=np.float64)
    source_positions = pd.Series(np.arange(len(source_frame)), index=source_frame.index)
    ref_groups = {
        key: group
        for key, group in reference_frame.groupby(list(group_cols), sort=False)
    }
    for key, src_group in source_frame.groupby(list(group_cols), sort=False):
        src_positions_group = source_positions.loc[src_group.index].to_numpy(dtype=np.int64)
        ref_group = ref_groups.get(key)
        if ref_group is None or ref_group.empty:
            continue

        src_r = src_group["r"].to_numpy(dtype=np.float64, copy=False)
        src_phi = src_group["phi"].to_numpy(dtype=np.float64, copy=False)
        src_z = src_group["z"].to_numpy(dtype=np.float64, copy=False)
        ref_r = ref_group["r"].to_numpy(dtype=np.float64, copy=False)
        ref_phi = ref_group["phi"].to_numpy(dtype=np.float64, copy=False)
        ref_z = ref_group["z"].to_numpy(dtype=np.float64, copy=False)

        for chunk_start in range(0, len(src_group), chunk_size):
            chunk_stop = min(chunk_start + chunk_size, len(src_group))
            chunk_r = src_r[chunk_start:chunk_stop]
            chunk_phi = src_phi[chunk_start:chunk_stop]
            chunk_z = src_z[chunk_start:chunk_stop]

            delta_phi = _wrapped_delta_phi(chunk_phi[:, None], ref_phi[None, :])
            r_mid = 0.5 * (chunk_r[:, None] + ref_r[None, :])
            delta_rphi = r_mid * delta_phi
            delta_z = chunk_z[:, None] - ref_z[None, :]
            dist2 = np.square(delta_rphi) + np.square(delta_z)

            if exclude_self and source_frame is reference_frame:
                src_ids = src_group.index.to_numpy()[chunk_start:chunk_stop]
                ref_ids = ref_group.index.to_numpy()
                dist2[src_ids[:, None] == ref_ids[None, :]] = np.inf

            out[src_positions_group[chunk_start:chunk_stop]] = np.sqrt(np.min(dist2, axis=1))

    return out


def _same_layer_neighbor_counts_within_distance(
    frame: pd.DataFrame,
    *,
    distance_threshold: float,
    group_cols: tuple[str, ...] = ("volume_id", "layer_id"),
    chunk_size: int = 256,
) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=np.float64)

    out = np.zeros(len(frame), dtype=np.float64)
    frame_positions = pd.Series(np.arange(len(frame)), index=frame.index)
    threshold2 = float(distance_threshold) ** 2
    for _, group in frame.groupby(list(group_cols), sort=False):
        group_len = len(group)
        group_positions = frame_positions.loc[group.index].to_numpy(dtype=np.int64)
        if group_len <= 1:
            continue

        grp_r = group["r"].to_numpy(dtype=np.float64, copy=False)
        grp_phi = group["phi"].to_numpy(dtype=np.float64, copy=False)
        grp_z = group["z"].to_numpy(dtype=np.float64, copy=False)

        for chunk_start in range(0, group_len, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, group_len)
            chunk_r = grp_r[chunk_start:chunk_stop]
            chunk_phi = grp_phi[chunk_start:chunk_stop]
            chunk_z = grp_z[chunk_start:chunk_stop]

            delta_phi = _wrapped_delta_phi(chunk_phi[:, None], grp_phi[None, :])
            r_mid = 0.5 * (chunk_r[:, None] + grp_r[None, :])
            delta_rphi = r_mid * delta_phi
            delta_z = chunk_z[:, None] - grp_z[None, :]
            dist2 = np.square(delta_rphi) + np.square(delta_z)

            rows = np.arange(chunk_stop - chunk_start)
            cols = np.arange(chunk_start, chunk_stop)
            dist2[rows, cols] = np.inf
            out[group_positions[chunk_start:chunk_stop]] = np.sum(dist2 < threshold2, axis=1)

    return out


def _threshold_counts_above(sorted_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if sorted_scores.size == 0:
        return np.zeros_like(thresholds, dtype=np.float64)
    return sorted_scores.size - np.searchsorted(sorted_scores, thresholds, side="left")


def _collect_event_cause_diagnostics(
    cfg: dict,
    assignments: BandAssignments,
    gap_half_width: float,
    crowding_distance: float,
) -> tuple[pd.DataFrame, list[int]]:
    import h5py

    if assignments.hit_eval_path is None:
        raise RuntimeError(f"Cause diagnostics require hit filtering, but split '{assignments.split}' has no hit_eval_path.")

    dataset_post, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=True)
    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    hit_threshold = float(dataset_post.hit_filter_threshold)

    known_volume_ids: set[int] = set()
    known_volume_layer_keys: set[tuple[int, int]] = set()
    rows: list[dict[str, float | int | str]] = []

    print(f"[{assignments.split}] collecting event-level cause diagnostics")
    with h5py.File(assignments.hit_eval_path, "r") as hit_eval_file:
        for event_idx in range(len(assignments.event_indices)):
            hits_post, particles_post = dataset_post.load_event(event_idx)
            hits_truth, particles_truth = dataset_truth.load_event(event_idx)
            hit_scores = _read_hit_filter_scores(hit_eval_file, int(assignments.sample_ids[event_idx]))

            post_particle_ids = particles_post["particle_id"]
            truth_particle_ids = particles_truth["particle_id"]

            valid_hits_post = hits_post[hits_post["on_valid_particle"]]
            invalid_hits_post = hits_post[~hits_post["on_valid_particle"]]
            truth_hits_valid = hits_truth[hits_truth["on_valid_particle"]]
            truth_hits_on_surviving_tracks = hits_truth[hits_truth["particle_id"].isin(post_particle_ids)]
            truth_pt = particles_truth["pt"].to_numpy(dtype=np.float64, copy=False)
            truth_abs_eta = np.abs(particles_truth["eta"].to_numpy(dtype=np.float64, copy=False))
            truth_track_phi = particles_truth["phi"].to_numpy(dtype=np.float64, copy=False)
            truth_track_eta = particles_truth["eta"].to_numpy(dtype=np.float64, copy=False)
            truth_track_phi_nn_gaps = _cyclic_nearest_neighbor_gaps(truth_track_phi, period=2.0 * np.pi)
            truth_track_phi_nn_gaps = truth_track_phi_nn_gaps[np.isfinite(truth_track_phi_nn_gaps)]
            truth_track_delta_r_nn = _delta_r_nearest_neighbor(truth_track_eta, truth_track_phi)
            truth_track_delta_r_nn = truth_track_delta_r_nn[np.isfinite(truth_track_delta_r_nn)]
            truth_hit_phi_nn_same_layer = _grouped_phi_nearest_neighbor_gaps(truth_hits_valid, ("volume_id", "layer_id"))
            truth_hit_rphiz_nn_same_layer = _same_layer_min_rphiz_distances(truth_hits_valid, truth_hits_valid, exclude_self=True)
            truth_hit_rphiz_nn_same_layer = truth_hit_rphiz_nn_same_layer[np.isfinite(truth_hit_rphiz_nn_same_layer)]
            truth_hit_neighbor_counts_same_layer = _same_layer_neighbor_counts_within_distance(
                truth_hits_valid,
                distance_threshold=crowding_distance,
            )
            truth_volume_layer_counts = (
                truth_hits_valid.groupby(["volume_id", "layer_id"]).size().to_numpy(dtype=np.int32)
                if not truth_hits_valid.empty
                else np.array([], dtype=np.int32)
            )

            post_hit_counts = (
                valid_hits_post.groupby("particle_id")
                .size()
                .reindex(post_particle_ids, fill_value=0)
                .to_numpy(dtype=np.int32)
            )
            pre_hit_counts_surviving = (
                truth_hits_on_surviving_tracks.groupby("particle_id")
                .size()
                .reindex(post_particle_ids, fill_value=0)
                .to_numpy(dtype=np.int32)
            )
            truth_hit_counts = (
                truth_hits_valid.groupby("particle_id")
                .size()
                .reindex(truth_particle_ids, fill_value=0)
                .to_numpy(dtype=np.int32)
            )
            surviving_truth_mask = truth_particle_ids.isin(post_particle_ids).to_numpy(dtype=bool)
            lost_truth_hit_counts = truth_hit_counts[~surviving_truth_mask]

            total_hits_post = int(len(hits_post))
            total_valid_hits_post = int(len(valid_hits_post))
            total_invalid_hits_post = int(len(invalid_hits_post))
            reco_pre = int(len(particles_truth))
            reco_post = int(len(particles_post))
            track_hit_retention = np.divide(
                post_hit_counts,
                pre_hit_counts_surviving,
                out=np.zeros_like(post_hit_counts, dtype=np.float64),
                where=pre_hit_counts_surviving > 0,
            )
            removed_true_hits = pre_hit_counts_surviving - post_hit_counts
            hit_shortfall = pre_hit_counts_surviving - post_hit_counts
            invalid_hit_particle_ids = invalid_hits_post["particle_id"].to_numpy(dtype=np.int64, copy=False)
            invalid_hit_on_truth_track = invalid_hit_particle_ids != 0
            invalid_hit_distance_to_true = _same_layer_min_rphiz_distances(invalid_hits_post, truth_hits_valid)
            invalid_hit_distance_to_true = invalid_hit_distance_to_true[np.isfinite(invalid_hit_distance_to_true)]
            invalid_hit_is_noise = invalid_hit_particle_ids == 0
            invalid_hit_near_true_mask = np.zeros(len(invalid_hits_post), dtype=bool)
            if len(invalid_hits_post) > 0:
                invalid_hit_min_distance_full = _same_layer_min_rphiz_distances(invalid_hits_post, truth_hits_valid)
                invalid_hit_near_true_mask = np.isfinite(invalid_hit_min_distance_full) & (invalid_hit_min_distance_full < crowding_distance)
            invalid_hit_nonreco_far_mask = (~invalid_hit_is_noise) & (~invalid_hit_near_true_mask)
            invalid_hit_noise_far_mask = invalid_hit_is_noise & (~invalid_hit_near_true_mask)

            total_volume_counts = hits_post["volume_id"].value_counts().to_dict()
            invalid_volume_counts = invalid_hits_post["volume_id"].value_counts().to_dict()
            truth_surviving_volume_counts = truth_hits_on_surviving_tracks["volume_id"].value_counts().to_dict()
            valid_post_volume_counts = valid_hits_post["volume_id"].value_counts().to_dict()
            total_volume_layer_counts = hits_post.groupby(["volume_id", "layer_id"]).size().to_dict()
            invalid_volume_layer_counts = invalid_hits_post.groupby(["volume_id", "layer_id"]).size().to_dict()
            truth_surviving_volume_layer_counts = truth_hits_on_surviving_tracks.groupby(["volume_id", "layer_id"]).size().to_dict()
            valid_post_volume_layer_counts = valid_hits_post.groupby(["volume_id", "layer_id"]).size().to_dict()
            known_volume_ids.update(int(v) for v in total_volume_counts)
            known_volume_layer_keys.update((int(v), int(l)) for v, l in total_volume_layer_counts)

            row: dict[str, float | int | str] = {
                "split": assignments.split,
                "event_index": int(assignments.event_indices[event_idx]),
                "sample_id": int(assignments.sample_ids[event_idx]),
                "event_name": assignments.event_names[event_idx],
                "band": BAND_NAMES[int(assignments.band_codes[event_idx])],
                "band_position": float(assignments.band_position[event_idx]),
                "upper_band_posterior": float(assignments.upper_band_posterior[event_idx]),
                "position_group": _event_position_group(float(assignments.band_position[event_idx]), gap_half_width=gap_half_width),
                "hits_after_filter": total_hits_post,
                "hits_on_reconstructable_tracks_after_filter": total_valid_hits_post,
                "hits_off_reconstructable_tracks_after_filter": total_invalid_hits_post,
                "filtered_hit_purity": _safe_fraction(total_valid_hits_post, total_hits_post),
                "filtered_hit_contamination": _safe_fraction(total_invalid_hits_post, total_hits_post),
                "off_track_hit_fraction_on_truth_particles": _safe_fraction(int(np.sum(invalid_hit_on_truth_track)), total_invalid_hits_post),
                "off_track_hit_fraction_noise": _safe_fraction(int(np.sum(~invalid_hit_on_truth_track)), total_invalid_hits_post),
                "reconstructable_tracks_pre_filter": reco_pre,
                "reconstructable_tracks_post_filter": reco_post,
                "lost_reconstructable_tracks": reco_pre - reco_post,
                "reconstructable_track_retention": _safe_fraction(reco_post, reco_pre),
                "track_pt_mean_pre": _mean_or_nan(truth_pt),
                "track_pt_median_pre": _percentile_or_nan(truth_pt, 50),
                "track_pt_p10_pre": _percentile_or_nan(truth_pt, 10),
                "track_pt_p90_pre": _percentile_or_nan(truth_pt, 90),
                "track_pt_sum_pre": float(np.sum(truth_pt.astype(np.float64))) if truth_pt.size > 0 else float("nan"),
                "fraction_tracks_pt_lt_2_pre": _safe_fraction(int(np.sum(truth_pt < 2.0)), reco_pre),
                "fraction_tracks_pt_lt_5_pre": _safe_fraction(int(np.sum(truth_pt < 5.0)), reco_pre),
                "track_abs_eta_mean_pre": _mean_or_nan(truth_abs_eta),
                "fraction_tracks_abs_eta_gt_2_pre": _safe_fraction(int(np.sum(truth_abs_eta > 2.0)), reco_pre),
                "num_occupied_volume_layers_true": int(truth_volume_layer_counts.size),
                "mean_true_hits_per_occupied_volume_layer": _mean_or_nan(truth_volume_layer_counts),
                "max_true_hits_single_volume_layer": float(np.max(truth_volume_layer_counts)) if truth_volume_layer_counts.size > 0 else np.nan,
                "mean_true_hit_phi_nn_gap_same_layer": _mean_or_nan(truth_hit_phi_nn_same_layer),
                "p10_true_hit_phi_nn_gap_same_layer": _percentile_or_nan(truth_hit_phi_nn_same_layer, 10),
                "fraction_true_hits_with_phi_nn_lt_0p02_same_layer": _safe_fraction(
                    int(np.sum(truth_hit_phi_nn_same_layer < 0.02)),
                    int(truth_hit_phi_nn_same_layer.size),
                ),
                "mean_true_hit_rphiz_nn_same_layer": _mean_or_nan(truth_hit_rphiz_nn_same_layer),
                "p10_true_hit_rphiz_nn_same_layer": _percentile_or_nan(truth_hit_rphiz_nn_same_layer, 10),
                "fraction_true_hits_with_close_rphiz_nn_same_layer": _safe_fraction(
                    int(np.sum(truth_hit_rphiz_nn_same_layer < crowding_distance)),
                    int(truth_hit_rphiz_nn_same_layer.size),
                ),
                "mean_true_hit_neighbors_within_crowding_distance_same_layer": _mean_or_nan(truth_hit_neighbor_counts_same_layer),
                "fraction_true_hits_with_5plus_neighbors_same_layer": _safe_fraction(
                    int(np.sum(truth_hit_neighbor_counts_same_layer >= 5)),
                    int(truth_hit_neighbor_counts_same_layer.size),
                ),
                "mean_true_track_phi_nn_gap": _mean_or_nan(truth_track_phi_nn_gaps),
                "p10_true_track_phi_nn_gap": _percentile_or_nan(truth_track_phi_nn_gaps, 10),
                "fraction_true_tracks_with_phi_nn_lt_0p05": _safe_fraction(
                    int(np.sum(truth_track_phi_nn_gaps < 0.05)),
                    int(truth_track_phi_nn_gaps.size),
                ),
                "mean_true_track_delta_r_nn": _mean_or_nan(truth_track_delta_r_nn),
                "p10_true_track_delta_r_nn": _percentile_or_nan(truth_track_delta_r_nn, 10),
                "fraction_true_tracks_with_delta_r_nn_lt_0p10": _safe_fraction(
                    int(np.sum(truth_track_delta_r_nn < 0.10)),
                    int(truth_track_delta_r_nn.size),
                ),
                "mean_pre_hits_per_surviving_track": _mean_or_nan(pre_hit_counts_surviving),
                "mean_post_hits_per_reconstructable_track": _mean_or_nan(post_hit_counts),
                "mean_removed_true_hits_per_surviving_track": _mean_or_nan(removed_true_hits),
                "mean_track_hit_retention": _mean_or_nan(track_hit_retention),
                "mean_hit_shortfall_surviving_track": _mean_or_nan(hit_shortfall),
                "event_hit_retention_on_surviving_tracks": _safe_fraction(int(np.sum(post_hit_counts)), int(np.sum(pre_hit_counts_surviving))),
                "fraction_tracks_with_3_post_hits": _safe_fraction(_count_equal(post_hit_counts, 3), reco_post),
                "fraction_tracks_with_4_post_hits": _safe_fraction(_count_equal(post_hit_counts, 4), reco_post),
                "fraction_tracks_with_5_post_hits": _safe_fraction(_count_equal(post_hit_counts, 5), reco_post),
                "fraction_tracks_with_6_post_hits": _safe_fraction(_count_equal(post_hit_counts, 6), reco_post),
                "fraction_tracks_with_7_post_hits": _safe_fraction(_count_equal(post_hit_counts, 7), reco_post),
                "fraction_tracks_with_8plus_post_hits": _safe_fraction(int(np.sum(post_hit_counts >= 8)), reco_post),
                "fraction_tracks_with_4_or_less_post_hits": _safe_fraction(int(np.sum(post_hit_counts <= 4)), reco_post),
                "fraction_surviving_tracks_lost_0_hits": _safe_fraction(_count_equal(hit_shortfall, 0), reco_post),
                "fraction_surviving_tracks_lost_1_hit": _safe_fraction(_count_equal(hit_shortfall, 1), reco_post),
                "fraction_surviving_tracks_lost_2plus_hits": _safe_fraction(int(np.sum(hit_shortfall >= 2)), reco_post),
                "lost_tracks_with_3_pre_hits": _count_equal(lost_truth_hit_counts, 3),
                "lost_tracks_with_4_pre_hits": _count_equal(lost_truth_hit_counts, 4),
                "lost_tracks_with_5_pre_hits": _count_equal(lost_truth_hit_counts, 5),
                "lost_tracks_with_6plus_pre_hits": int(np.sum(lost_truth_hit_counts >= 6)),
                "mean_invalid_hit_distance_to_true_same_layer": _mean_or_nan(invalid_hit_distance_to_true),
                "kept_invalid_hits_near_true_same_layer": int(np.sum(invalid_hit_near_true_mask)),
                "kept_invalid_hits_on_nonreco_truth_far": int(np.sum(invalid_hit_nonreco_far_mask)),
                "kept_invalid_hits_noise_far": int(np.sum(invalid_hit_noise_far_mask)),
                "fraction_invalid_hits_near_true_same_layer": _safe_fraction(int(np.sum(invalid_hit_near_true_mask)), total_invalid_hits_post),
                "fraction_invalid_hits_on_nonreco_truth_far": _safe_fraction(int(np.sum(invalid_hit_nonreco_far_mask)), total_invalid_hits_post),
                "fraction_invalid_hits_noise_far": _safe_fraction(int(np.sum(invalid_hit_noise_far_mask)), total_invalid_hits_post),
            }

            if hit_scores is not None and len(hit_scores) == len(hits_truth):
                truth_valid_mask = hits_truth["on_valid_particle"].to_numpy(dtype=bool)
                valid_scores = hit_scores[truth_valid_mask]
                invalid_scores = hit_scores[~truth_valid_mask]
                row.update(
                    {
                        "score_mean_all_pre": _mean_or_nan(hit_scores),
                        "score_std_all_pre": float(np.std(hit_scores.astype(np.float64))),
                        "score_p10_all_pre": _percentile_or_nan(hit_scores, 10),
                        "score_p50_all_pre": _percentile_or_nan(hit_scores, 50),
                        "score_p90_all_pre": _percentile_or_nan(hit_scores, 90),
                        "score_mean_valid_pre": _mean_or_nan(valid_scores),
                        "score_mean_invalid_pre": _mean_or_nan(invalid_scores),
                        "score_mean_valid_minus_invalid_pre": _mean_or_nan(valid_scores) - _mean_or_nan(invalid_scores),
                        "score_fraction_valid_above_threshold_pre": _safe_fraction(int(np.sum(valid_scores >= hit_threshold)), int(valid_scores.size)),
                        "score_fraction_invalid_above_threshold_pre": _safe_fraction(int(np.sum(invalid_scores >= hit_threshold)), int(invalid_scores.size)),
                        "score_fraction_valid_near_threshold_pre": _safe_fraction(
                            int(np.sum(np.abs(valid_scores - hit_threshold) <= 0.05)),
                            int(valid_scores.size),
                        ),
                        "score_fraction_invalid_near_threshold_pre": _safe_fraction(
                            int(np.sum(np.abs(invalid_scores - hit_threshold) <= 0.05)),
                            int(invalid_scores.size),
                        ),
                        "score_margin_fraction_pre": _safe_fraction(int(np.sum(np.abs(hit_scores - hit_threshold) <= 0.05)), int(hit_scores.size)),
                    }
                )
            else:
                row.update(
                    {
                        "score_mean_all_pre": np.nan,
                        "score_std_all_pre": np.nan,
                        "score_p10_all_pre": np.nan,
                        "score_p50_all_pre": np.nan,
                        "score_p90_all_pre": np.nan,
                        "score_mean_valid_pre": np.nan,
                        "score_mean_invalid_pre": np.nan,
                        "score_mean_valid_minus_invalid_pre": np.nan,
                        "score_fraction_valid_above_threshold_pre": np.nan,
                        "score_fraction_invalid_above_threshold_pre": np.nan,
                        "score_fraction_valid_near_threshold_pre": np.nan,
                        "score_fraction_invalid_near_threshold_pre": np.nan,
                        "score_margin_fraction_pre": np.nan,
                    }
                )

            for volume_id, count in total_volume_counts.items():
                row[f"hit_fraction_volume_{int(volume_id)}"] = _safe_fraction(int(count), total_hits_post)
            for volume_id, count in invalid_volume_counts.items():
                row[f"invalid_hit_fraction_volume_{int(volume_id)}"] = _safe_fraction(int(count), total_invalid_hits_post)
            for volume_id, pre_count in truth_surviving_volume_counts.items():
                post_count = int(valid_post_volume_counts.get(volume_id, 0))
                row[f"retention_volume_{int(volume_id)}"] = _safe_fraction(post_count, int(pre_count))
            for (volume_id, layer_id), count in total_volume_layer_counts.items():
                row[f"hit_fraction_volume_{int(volume_id)}_layer_{int(layer_id)}"] = _safe_fraction(int(count), total_hits_post)
            for (volume_id, layer_id), count in invalid_volume_layer_counts.items():
                row[f"invalid_hit_fraction_volume_{int(volume_id)}_layer_{int(layer_id)}"] = _safe_fraction(int(count), total_invalid_hits_post)
            for (volume_id, layer_id), pre_count in truth_surviving_volume_layer_counts.items():
                post_count = int(valid_post_volume_layer_counts.get((volume_id, layer_id), 0))
                row[f"retention_volume_{int(volume_id)}_layer_{int(layer_id)}"] = _safe_fraction(post_count, int(pre_count))

            rows.append(row)

            if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.event_indices):
                print(f"[{assignments.split}] event diagnostics processed {event_idx + 1}/{len(assignments.event_indices)}")

    volume_ids = sorted(known_volume_ids)
    volume_layer_keys = sorted(known_volume_layer_keys)
    df = pd.DataFrame(rows)
    for col in _volume_fraction_columns(volume_ids, "hit_fraction"):
        if col not in df.columns:
            df[col] = 0.0
    for col in _volume_fraction_columns(volume_ids, "invalid_hit_fraction"):
        if col not in df.columns:
            df[col] = 0.0
    for col in _volume_fraction_columns(volume_ids, "retention"):
        if col not in df.columns:
            df[col] = np.nan
    for col in _volume_layer_fraction_columns(volume_layer_keys, "hit_fraction"):
        if col not in df.columns:
            df[col] = 0.0
    for col in _volume_layer_fraction_columns(volume_layer_keys, "invalid_hit_fraction"):
        if col not in df.columns:
            df[col] = 0.0
    for col in _volume_layer_fraction_columns(volume_layer_keys, "retention"):
        if col not in df.columns:
            df[col] = np.nan

    ordered_cols = [
        "split",
        "event_index",
        "sample_id",
        "event_name",
        "band",
        "band_position",
        "upper_band_posterior",
        "position_group",
        "hits_after_filter",
        "hits_on_reconstructable_tracks_after_filter",
        "hits_off_reconstructable_tracks_after_filter",
        "filtered_hit_purity",
        "filtered_hit_contamination",
        "off_track_hit_fraction_on_truth_particles",
        "off_track_hit_fraction_noise",
        "reconstructable_tracks_pre_filter",
        "reconstructable_tracks_post_filter",
        "lost_reconstructable_tracks",
        "reconstructable_track_retention",
        "track_pt_mean_pre",
        "track_pt_median_pre",
        "track_pt_p10_pre",
        "track_pt_p90_pre",
        "track_pt_sum_pre",
        "fraction_tracks_pt_lt_2_pre",
        "fraction_tracks_pt_lt_5_pre",
        "track_abs_eta_mean_pre",
        "fraction_tracks_abs_eta_gt_2_pre",
        "num_occupied_volume_layers_true",
        "mean_true_hits_per_occupied_volume_layer",
        "max_true_hits_single_volume_layer",
        "mean_true_hit_phi_nn_gap_same_layer",
        "p10_true_hit_phi_nn_gap_same_layer",
        "fraction_true_hits_with_phi_nn_lt_0p02_same_layer",
        "mean_true_hit_rphiz_nn_same_layer",
        "p10_true_hit_rphiz_nn_same_layer",
        "fraction_true_hits_with_close_rphiz_nn_same_layer",
        "mean_true_hit_neighbors_within_crowding_distance_same_layer",
        "fraction_true_hits_with_5plus_neighbors_same_layer",
        "mean_true_track_phi_nn_gap",
        "p10_true_track_phi_nn_gap",
        "fraction_true_tracks_with_phi_nn_lt_0p05",
        "mean_true_track_delta_r_nn",
        "p10_true_track_delta_r_nn",
        "fraction_true_tracks_with_delta_r_nn_lt_0p10",
        "mean_pre_hits_per_surviving_track",
        "mean_post_hits_per_reconstructable_track",
        "mean_removed_true_hits_per_surviving_track",
        "mean_track_hit_retention",
        "mean_hit_shortfall_surviving_track",
        "event_hit_retention_on_surviving_tracks",
        "fraction_tracks_with_3_post_hits",
        "fraction_tracks_with_4_post_hits",
        "fraction_tracks_with_5_post_hits",
        "fraction_tracks_with_6_post_hits",
        "fraction_tracks_with_7_post_hits",
        "fraction_tracks_with_8plus_post_hits",
        "fraction_tracks_with_4_or_less_post_hits",
        "fraction_surviving_tracks_lost_0_hits",
        "fraction_surviving_tracks_lost_1_hit",
        "fraction_surviving_tracks_lost_2plus_hits",
        "lost_tracks_with_3_pre_hits",
        "lost_tracks_with_4_pre_hits",
        "lost_tracks_with_5_pre_hits",
        "lost_tracks_with_6plus_pre_hits",
        "mean_invalid_hit_distance_to_true_same_layer",
        "kept_invalid_hits_near_true_same_layer",
        "kept_invalid_hits_on_nonreco_truth_far",
        "kept_invalid_hits_noise_far",
        "fraction_invalid_hits_near_true_same_layer",
        "fraction_invalid_hits_on_nonreco_truth_far",
        "fraction_invalid_hits_noise_far",
        "score_mean_all_pre",
        "score_std_all_pre",
        "score_p10_all_pre",
        "score_p50_all_pre",
        "score_p90_all_pre",
        "score_mean_valid_pre",
        "score_mean_invalid_pre",
        "score_mean_valid_minus_invalid_pre",
        "score_fraction_valid_above_threshold_pre",
        "score_fraction_invalid_above_threshold_pre",
        "score_fraction_valid_near_threshold_pre",
        "score_fraction_invalid_near_threshold_pre",
        "score_margin_fraction_pre",
    ] + _volume_fraction_columns(volume_ids, "hit_fraction") + _volume_fraction_columns(volume_ids, "invalid_hit_fraction") + _volume_fraction_columns(volume_ids, "retention") + _volume_layer_fraction_columns(volume_layer_keys, "hit_fraction") + _volume_layer_fraction_columns(volume_layer_keys, "invalid_hit_fraction") + _volume_layer_fraction_columns(volume_layer_keys, "retention")
    return df.loc[:, ordered_cols], volume_ids


def _write_event_cause_diagnostics_csv(out_path: Path, diagnostics: pd.DataFrame) -> None:
    diagnostics.to_csv(out_path, index=False)


def _write_event_cause_summary_csv(out_path: Path, diagnostics: pd.DataFrame) -> None:
    metric_columns = [
        "band_position",
        "upper_band_posterior",
        "hits_after_filter",
        "hits_on_reconstructable_tracks_after_filter",
        "hits_off_reconstructable_tracks_after_filter",
        "filtered_hit_purity",
        "filtered_hit_contamination",
        "off_track_hit_fraction_on_truth_particles",
        "off_track_hit_fraction_noise",
        "reconstructable_tracks_pre_filter",
        "reconstructable_tracks_post_filter",
        "lost_reconstructable_tracks",
        "reconstructable_track_retention",
        "track_pt_mean_pre",
        "track_pt_median_pre",
        "track_pt_p10_pre",
        "track_pt_p90_pre",
        "track_pt_sum_pre",
        "fraction_tracks_pt_lt_2_pre",
        "fraction_tracks_pt_lt_5_pre",
        "track_abs_eta_mean_pre",
        "fraction_tracks_abs_eta_gt_2_pre",
        "num_occupied_volume_layers_true",
        "mean_true_hits_per_occupied_volume_layer",
        "max_true_hits_single_volume_layer",
        "mean_true_hit_phi_nn_gap_same_layer",
        "p10_true_hit_phi_nn_gap_same_layer",
        "fraction_true_hits_with_phi_nn_lt_0p02_same_layer",
        "mean_true_hit_rphiz_nn_same_layer",
        "p10_true_hit_rphiz_nn_same_layer",
        "fraction_true_hits_with_close_rphiz_nn_same_layer",
        "mean_true_hit_neighbors_within_crowding_distance_same_layer",
        "fraction_true_hits_with_5plus_neighbors_same_layer",
        "mean_true_track_phi_nn_gap",
        "p10_true_track_phi_nn_gap",
        "fraction_true_tracks_with_phi_nn_lt_0p05",
        "mean_true_track_delta_r_nn",
        "p10_true_track_delta_r_nn",
        "fraction_true_tracks_with_delta_r_nn_lt_0p10",
        "mean_pre_hits_per_surviving_track",
        "mean_post_hits_per_reconstructable_track",
        "mean_removed_true_hits_per_surviving_track",
        "mean_track_hit_retention",
        "mean_hit_shortfall_surviving_track",
        "event_hit_retention_on_surviving_tracks",
        "fraction_tracks_with_3_post_hits",
        "fraction_tracks_with_4_post_hits",
        "fraction_tracks_with_5_post_hits",
        "fraction_tracks_with_6_post_hits",
        "fraction_tracks_with_4_or_less_post_hits",
        "fraction_surviving_tracks_lost_0_hits",
        "fraction_surviving_tracks_lost_1_hit",
        "fraction_surviving_tracks_lost_2plus_hits",
        "lost_tracks_with_3_pre_hits",
        "lost_tracks_with_4_pre_hits",
        "lost_tracks_with_5_pre_hits",
        "lost_tracks_with_6plus_pre_hits",
        "mean_invalid_hit_distance_to_true_same_layer",
        "kept_invalid_hits_near_true_same_layer",
        "kept_invalid_hits_on_nonreco_truth_far",
        "kept_invalid_hits_noise_far",
        "fraction_invalid_hits_near_true_same_layer",
        "fraction_invalid_hits_on_nonreco_truth_far",
        "fraction_invalid_hits_noise_far",
        "score_mean_all_pre",
        "score_std_all_pre",
        "score_mean_valid_pre",
        "score_mean_invalid_pre",
        "score_mean_valid_minus_invalid_pre",
        "score_fraction_valid_above_threshold_pre",
        "score_fraction_invalid_above_threshold_pre",
        "score_fraction_valid_near_threshold_pre",
        "score_fraction_invalid_near_threshold_pre",
    ]
    group_order = list(POSITION_GROUPS)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position_group", "num_events", "metric", "mean", "median", "std"])
        for group_name in group_order:
            group_df = diagnostics[diagnostics["position_group"] == group_name]
            for metric in metric_columns:
                values = group_df[metric].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    writer.writerow([group_name, int(len(group_df)), metric, np.nan, np.nan, np.nan])
                    continue
                writer.writerow(
                    [
                        group_name,
                        int(len(group_df)),
                        metric,
                        float(np.mean(values)),
                        float(np.median(values)),
                        float(np.std(values)),
                    ]
                )


def _plot_event_cause_diagnostics(diagnostics: pd.DataFrame, output_base: Path, gap_half_width: float) -> None:
    plt = _get_pyplot()

    fig, axes = plt.subplots(5, 3, figsize=(14.8, 17.2), constrained_layout=False)
    fig.subplots_adjust(top=0.93, hspace=0.42, wspace=0.30)
    axes = axes.ravel()
    position = diagnostics["band_position"].to_numpy(dtype=np.float64)
    group_names = ["lower_core", "between_bands", "upper_core"]
    group_colors = {
        "lower_core": BAND_COLORS["lower_band"],
        "between_bands": "#666666",
        "upper_core": BAND_COLORS["upper_band"],
    }

    def _scatter_panel(ax, y_col: str, title: str, ylabel: str) -> None:
        for group_name in group_names:
            mask = diagnostics["position_group"] == group_name
            ax.scatter(
                diagnostics.loc[mask, "band_position"],
                diagnostics.loc[mask, y_col],
                s=10.0,
                alpha=0.55,
                color=group_colors[group_name],
                edgecolors="none",
                rasterized=True,
                label=group_name.replace("_", " "),
            )
        ax.axvspan(0.5 - gap_half_width, 0.5 + gap_half_width, color="#d9d9d9", alpha=0.25)
        ax.set_xlabel("Position between fitted bands")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    bins = np.linspace(np.nanmin(position), np.nanmax(position), 50) if len(position) > 1 else np.linspace(0.0, 1.0, 50)
    axes[0].hist(
        position,
        bins=bins,
        weights=_hist_weights(len(position), num_events=len(position), allow_count_per_event=True),
        density=_hist_density(),
        color="#4c78a8",
        histtype="step",
        linewidth=1.4,
    )
    axes[0].axvspan(0.5 - gap_half_width, 0.5 + gap_half_width, color="#d9d9d9", alpha=0.25)
    axes[0].set_title("Event positions between fitted bands")
    axes[0].set_xlabel("Position between fitted bands")
    axes[0].set_ylabel(_event_hist_ylabel())
    axes[0].grid(alpha=0.10, linestyle="--", linewidth=0.45)
    axes[0].set_axisbelow(True)

    posterior = diagnostics["upper_band_posterior"].to_numpy(dtype=np.float64)
    posterior = posterior[np.isfinite(posterior)]
    axes[1].hist(
        posterior,
        bins=np.linspace(0.0, 1.0, 41),
        weights=_hist_weights(len(posterior), num_events=len(posterior), allow_count_per_event=True),
        density=_hist_density(),
        color="#4c78a8",
        histtype="step",
        linewidth=1.4,
    )
    axes[1].set_title("Assignment confidence")
    axes[1].set_xlabel("Posterior P(upper band)")
    axes[1].set_ylabel(_event_hist_ylabel())
    axes[1].grid(alpha=0.10, linestyle="--", linewidth=0.45)
    axes[1].set_axisbelow(True)

    _scatter_panel(
        axes[2],
        y_col="reconstructable_tracks_pre_filter",
        title="Reconstructable tracks per event vs band position",
        ylabel="Reconstructable tracks pre filter",
    )

    _scatter_panel(
        axes[3],
        y_col="mean_pre_hits_per_surviving_track",
        title="Pre-filter hits per surviving track vs band position",
        ylabel="Mean pre hits / surviving track",
    )

    _scatter_panel(
        axes[4],
        y_col="track_pt_mean_pre",
        title="Mean track pT vs band position",
        ylabel="Mean track pT [GeV]",
    )

    _scatter_panel(
        axes[5],
        y_col="fraction_tracks_pt_lt_2_pre",
        title="Low-pT track fraction vs band position",
        ylabel="Fraction with pT < 2 GeV",
    )

    _scatter_panel(
        axes[6],
        y_col="mean_true_hits_per_occupied_volume_layer",
        title="True-hit density per occupied layer vs band position",
        ylabel="Mean true hits / occupied volume-layer",
    )

    _scatter_panel(
        axes[7],
        y_col="fraction_true_hits_with_close_rphiz_nn_same_layer",
        title="Close true-hit neighbors vs band position",
        ylabel="Fraction with close same-layer rphi/z NN",
    )

    _scatter_panel(
        axes[8],
        y_col="fraction_true_tracks_with_delta_r_nn_lt_0p10",
        title="Close truth-track neighbors vs band position",
        ylabel="Fraction with track deltaR NN < 0.10",
    )

    _scatter_panel(
        axes[9],
        y_col="hits_off_reconstructable_tracks_after_filter",
        title="Off-track filtered hits vs band position",
        ylabel="Hits off reconstructable tracks",
    )
    _scatter_panel(
        axes[10],
        y_col="filtered_hit_contamination",
        title="Filtered-hit contamination vs band position",
        ylabel="Contamination fraction",
    )
    _scatter_panel(
        axes[11],
        y_col="score_mean_valid_minus_invalid_pre",
        title="Valid-invalid score gap vs band position",
        ylabel="Mean valid score - invalid score",
    )
    _scatter_panel(
        axes[12],
        y_col="fraction_invalid_hits_near_true_same_layer",
        title="Invalid hits near true hits vs band position",
        ylabel="Fraction of invalid hits near true hits",
    )
    _scatter_panel(
        axes[13],
        y_col="fraction_invalid_hits_on_nonreco_truth_far",
        title="Invalid nonreco-truth hits vs band position",
        ylabel="Fraction of invalid hits on nonreco truth",
    )
    _scatter_panel(
        axes[14],
        y_col="fraction_invalid_hits_noise_far",
        title="Invalid noise hits vs band position",
        ylabel="Fraction of invalid hits on noise",
    )

    fig.suptitle("Cause diagnostics for post-filter band structure", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_volume_fraction_diagnostics(
    diagnostics: pd.DataFrame,
    volume_ids: list[int],
    output_base: Path,
) -> None:
    plt = _get_pyplot()
    group_names = ["lower_core", "between_bands", "upper_core"]
    group_colors = {
        "lower_core": BAND_COLORS["lower_band"],
        "between_bands": "#666666",
        "upper_core": BAND_COLORS["upper_band"],
    }
    x = np.arange(len(volume_ids), dtype=np.float64)
    width = 0.24

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.2), constrained_layout=True)
    panels = [
        ("hit_fraction", "Filtered hit fraction by volume", "Mean fraction of filtered hits"),
        ("retention", "True-hit retention by volume on surviving tracks", "Mean retained-hit fraction"),
        ("invalid_hit_fraction", "Off-track filtered hit fraction by volume", "Mean fraction of off-track hits"),
    ]
    for ax, (prefix, title, ylabel) in zip(axes, panels, strict=True):
        for i, group_name in enumerate(group_names):
            group_df = diagnostics[diagnostics["position_group"] == group_name]
            means = [
                float(group_df[f"{prefix}_volume_{volume_id}"].mean()) if len(group_df) > 0 else 0.0
                for volume_id in volume_ids
            ]
            ax.bar(
                x + (i - 1) * width,
                means,
                width=width,
                label=group_name.replace("_", " "),
                color=group_colors[group_name],
                alpha=0.85,
            )
        ax.set_xticks(x, [str(v) for v in volume_ids])
        ax.set_xlabel("Volume ID")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="y")
        ax.set_axisbelow(True)

    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Detector-volume diagnostics for post-filter band structure", fontsize=13, y=0.995)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _collect_band_hitfilter_diagnostics(
    cfg: dict,
    assignments: BandAssignments,
    *,
    max_score_samples_per_band: int,
    rng_seed: int,
    num_threshold_points: int,
    crowding_distance: float,
) -> BandHitFilterDiagnostics:
    import h5py

    if assignments.hit_eval_path is None:
        raise RuntimeError(f"Hit-filter diagnostics require hit filtering, but split '{assignments.split}' has no hit_eval_path.")

    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    hit_threshold = float(cfg["data"].get("hit_filter_threshold", 0.1))
    min_track_hits = int(cfg["data"]["particle_min_num_hits"])
    thresholds = np.linspace(0.0, 1.0, max(2, num_threshold_points))
    rng = np.random.default_rng(rng_seed + 2000)

    score_samples_by_band = {
        band_name: pd.DataFrame(columns=["score", "truth_label", "invalid_category", "volume_id", "layer_id", "_sample_key"])
        for band_name in BAND_NAMES
    }
    sweep_sums = {
        band_name: {
            "num_events": 0,
            "kept_invalid_hits_per_event": np.zeros_like(thresholds, dtype=np.float64),
            "contamination_fraction": np.zeros_like(thresholds, dtype=np.float64),
            "kept_true_hits_per_event": np.zeros_like(thresholds, dtype=np.float64),
            "true_hit_recall": np.zeros_like(thresholds, dtype=np.float64),
            "reconstructable_tracks_per_event": np.zeros_like(thresholds, dtype=np.float64),
        }
        for band_name in BAND_NAMES
    }
    region_accumulators: dict[tuple[str, int, int], dict[str, float]] = {}
    invalid_category_counts = {
        band_name: {
            "near_true_same_layer": 0,
            "nonreco_truth_far": 0,
            "noise_far": 0,
            "total_kept_invalid": 0,
        }
        for band_name in BAND_NAMES
    }
    invalid_category_totals = {
        band_name: {
            "near_true_same_layer": 0,
            "nonreco_truth_far": 0,
            "noise_far": 0,
        }
        for band_name in BAND_NAMES
    }

    print(f"[{assignments.split}] collecting score, threshold-sweep, and region-conditioned diagnostics")
    with h5py.File(assignments.hit_eval_path, "r") as hit_eval_file:
        for event_idx, band_code in enumerate(assignments.band_codes):
            band_name = BAND_NAMES[int(band_code)]
            hits_truth, particles_truth = dataset_truth.load_event(event_idx)
            hit_scores = _read_hit_filter_scores(hit_eval_file, int(assignments.sample_ids[event_idx]))
            if hit_scores is None or len(hit_scores) != len(hits_truth):
                raise RuntimeError(
                    f"Missing or misaligned hit-filter scores for split '{assignments.split}', sample_id={assignments.sample_ids[event_idx]}."
                )

            scores = np.asarray(hit_scores, dtype=np.float64)
            truth_valid_mask = hits_truth["on_valid_particle"].to_numpy(dtype=bool)
            invalid_mask = ~truth_valid_mask
            valid_scores = scores[truth_valid_mask]
            invalid_scores = scores[invalid_mask]
            invalid_particle_ids = hits_truth.loc[invalid_mask, "particle_id"].to_numpy(dtype=np.int64, copy=False)

            truth_hits_valid = hits_truth.loc[truth_valid_mask, :]
            invalid_hits_truth = hits_truth.loc[invalid_mask, :]
            invalid_min_distances = _same_layer_min_rphiz_distances(invalid_hits_truth, truth_hits_valid)
            invalid_near_true = np.isfinite(invalid_min_distances) & (invalid_min_distances < crowding_distance)
            invalid_noise = invalid_particle_ids == 0
            invalid_category = np.where(
                invalid_near_true,
                "near_true_same_layer",
                np.where(invalid_noise, "noise_far", "nonreco_truth_far"),
            )
            for category in ("near_true_same_layer", "nonreco_truth_far", "noise_far"):
                invalid_category_totals[band_name][category] += int(np.sum(invalid_category == category))

            valid_sample = pd.DataFrame(
                {
                    "score": valid_scores,
                    "truth_label": "valid",
                    "invalid_category": "valid",
                    "volume_id": hits_truth.loc[truth_valid_mask, "volume_id"].to_numpy(dtype=np.int32, copy=False),
                    "layer_id": hits_truth.loc[truth_valid_mask, "layer_id"].to_numpy(dtype=np.int32, copy=False),
                }
            )
            invalid_sample = pd.DataFrame(
                {
                    "score": invalid_scores,
                    "truth_label": "invalid",
                    "invalid_category": invalid_category,
                    "volume_id": hits_truth.loc[invalid_mask, "volume_id"].to_numpy(dtype=np.int32, copy=False),
                    "layer_id": hits_truth.loc[invalid_mask, "layer_id"].to_numpy(dtype=np.int32, copy=False),
                }
            )
            score_samples_by_band[band_name] = _priority_sample(
                score_samples_by_band[band_name],
                pd.concat([valid_sample, invalid_sample], ignore_index=True),
                max_rows=max_score_samples_per_band,
                rng=rng,
            )

            valid_sorted = np.sort(valid_scores)
            invalid_sorted = np.sort(invalid_scores)
            kept_valid_counts = _threshold_counts_above(valid_sorted, thresholds)
            kept_invalid_counts = _threshold_counts_above(invalid_sorted, thresholds)
            kept_total_counts = kept_valid_counts + kept_invalid_counts
            contamination = np.divide(
                kept_invalid_counts,
                kept_total_counts,
                out=np.zeros_like(kept_invalid_counts, dtype=np.float64),
                where=kept_total_counts > 0,
            )
            recall = np.divide(
                kept_valid_counts,
                max(valid_scores.size, 1),
                out=np.zeros_like(kept_valid_counts, dtype=np.float64),
                where=valid_scores.size > 0,
            )

            valid_particle_ids = hits_truth.loc[truth_valid_mask, "particle_id"].to_numpy(dtype=np.int64, copy=False)
            if valid_particle_ids.size > 0:
                order = np.argsort(valid_particle_ids, kind="stable")
                valid_particle_ids_sorted = valid_particle_ids[order]
                valid_scores_by_particle = valid_scores[order]
                _, starts, counts = np.unique(valid_particle_ids_sorted, return_index=True, return_counts=True)
                kth_scores = np.empty(len(starts), dtype=np.float64)
                for i, (start, count) in enumerate(zip(starts, counts, strict=True)):
                    track_scores = np.sort(valid_scores_by_particle[start : start + count])
                    kth_scores[i] = track_scores[-min_track_hits]
                reco_counts = _threshold_counts_above(np.sort(kth_scores), thresholds)
            else:
                reco_counts = np.zeros_like(thresholds, dtype=np.float64)

            sweep_sums[band_name]["num_events"] += 1
            sweep_sums[band_name]["kept_invalid_hits_per_event"] += kept_invalid_counts
            sweep_sums[band_name]["contamination_fraction"] += contamination
            sweep_sums[band_name]["kept_true_hits_per_event"] += kept_valid_counts
            sweep_sums[band_name]["true_hit_recall"] += recall
            sweep_sums[band_name]["reconstructable_tracks_per_event"] += reco_counts

            kept_default = scores >= hit_threshold
            kept_invalid_default = invalid_mask & kept_default
            kept_invalid_near_true = int(np.sum(invalid_near_true & kept_default[invalid_mask]))
            kept_invalid_nonreco_truth_far = int(np.sum((invalid_category == "nonreco_truth_far") & kept_default[invalid_mask]))
            kept_invalid_noise_far = int(np.sum((invalid_category == "noise_far") & kept_default[invalid_mask]))

            invalid_category_counts[band_name]["near_true_same_layer"] += kept_invalid_near_true
            invalid_category_counts[band_name]["nonreco_truth_far"] += kept_invalid_nonreco_truth_far
            invalid_category_counts[band_name]["noise_far"] += kept_invalid_noise_far
            invalid_category_counts[band_name]["total_kept_invalid"] += int(np.sum(kept_invalid_default))

            region_frame = pd.DataFrame(
                {
                    "volume_id": hits_truth["volume_id"].to_numpy(dtype=np.int32, copy=False),
                    "layer_id": hits_truth["layer_id"].to_numpy(dtype=np.int32, copy=False),
                    "score": scores,
                    "truth_valid": truth_valid_mask,
                    "kept_default": kept_default,
                }
            )
            invalid_near_true_full = np.zeros(len(hits_truth), dtype=bool)
            invalid_near_true_full[invalid_mask] = invalid_near_true
            region_frame["invalid_near_true"] = invalid_near_true_full

            for (volume_id, layer_id), group in region_frame.groupby(["volume_id", "layer_id"], sort=False):
                key = (band_name, int(volume_id), int(layer_id))
                acc = region_accumulators.setdefault(
                    key,
                    {
                        "valid_hits": 0.0,
                        "invalid_hits": 0.0,
                        "kept_valid_hits": 0.0,
                        "kept_invalid_hits": 0.0,
                        "kept_invalid_near_true_hits": 0.0,
                        "valid_score_sum": 0.0,
                        "invalid_score_sum": 0.0,
                    },
                )
                valid_group_mask = group["truth_valid"].to_numpy(dtype=bool)
                invalid_group_mask = ~valid_group_mask
                kept_group_mask = group["kept_default"].to_numpy(dtype=bool)
                near_true_group_mask = group["invalid_near_true"].to_numpy(dtype=bool)
                score_group = group["score"].to_numpy(dtype=np.float64, copy=False)

                acc["valid_hits"] += float(np.sum(valid_group_mask))
                acc["invalid_hits"] += float(np.sum(invalid_group_mask))
                acc["kept_valid_hits"] += float(np.sum(valid_group_mask & kept_group_mask))
                acc["kept_invalid_hits"] += float(np.sum(invalid_group_mask & kept_group_mask))
                acc["kept_invalid_near_true_hits"] += float(np.sum(invalid_group_mask & kept_group_mask & near_true_group_mask))
                acc["valid_score_sum"] += float(np.sum(score_group[valid_group_mask]))
                acc["invalid_score_sum"] += float(np.sum(score_group[invalid_group_mask]))

            if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
                print(f"[{assignments.split}] hit-filter diagnostics processed {event_idx + 1}/{len(assignments.band_codes)}")

    threshold_rows: list[dict[str, float | str]] = []
    for band_name in BAND_NAMES:
        num_events = max(1, int(sweep_sums[band_name]["num_events"]))
        for i, threshold in enumerate(thresholds):
            threshold_rows.append(
                {
                    "split": assignments.split,
                    "band": band_name,
                    "threshold": float(threshold),
                    "num_events": num_events,
                    "mean_kept_invalid_hits_per_event": float(sweep_sums[band_name]["kept_invalid_hits_per_event"][i] / num_events),
                    "mean_contamination_fraction": float(sweep_sums[band_name]["contamination_fraction"][i] / num_events),
                    "mean_kept_true_hits_per_event": float(sweep_sums[band_name]["kept_true_hits_per_event"][i] / num_events),
                    "mean_true_hit_recall": float(sweep_sums[band_name]["true_hit_recall"][i] / num_events),
                    "mean_reconstructable_tracks_per_event": float(sweep_sums[band_name]["reconstructable_tracks_per_event"][i] / num_events),
                }
            )

    region_rows: list[dict[str, float | int | str]] = []
    for (band_name, volume_id, layer_id), acc in region_accumulators.items():
        kept_total = acc["kept_valid_hits"] + acc["kept_invalid_hits"]
        mean_valid_score = float(acc["valid_score_sum"] / acc["valid_hits"]) if acc["valid_hits"] > 0 else np.nan
        mean_invalid_score = float(acc["invalid_score_sum"] / acc["invalid_hits"]) if acc["invalid_hits"] > 0 else np.nan
        region_rows.append(
            {
                "split": assignments.split,
                "band": band_name,
                "volume_id": volume_id,
                "layer_id": layer_id,
                "valid_hits": acc["valid_hits"],
                "invalid_hits": acc["invalid_hits"],
                "kept_valid_hits": acc["kept_valid_hits"],
                "kept_invalid_hits": acc["kept_invalid_hits"],
                "kept_contamination_fraction": float(acc["kept_invalid_hits"] / kept_total) if kept_total > 0 else np.nan,
                "kept_invalid_near_true_fraction": float(acc["kept_invalid_near_true_hits"] / max(acc["kept_invalid_hits"], 1.0)),
                "mean_valid_score_pre": mean_valid_score,
                "mean_invalid_score_pre": mean_invalid_score,
                "score_gap_pre": mean_valid_score - mean_invalid_score if np.isfinite(mean_valid_score) and np.isfinite(mean_invalid_score) else np.nan,
            }
        )

    invalid_category_rows: list[dict[str, float | int | str]] = []
    for band_name in BAND_NAMES:
        total_kept_invalid = max(1, invalid_category_counts[band_name]["total_kept_invalid"])
        for category in ("near_true_same_layer", "nonreco_truth_far", "noise_far"):
            count = int(invalid_category_counts[band_name][category])
            invalid_category_rows.append(
                {
                    "split": assignments.split,
                    "band": band_name,
                    "category": category,
                    "count": count,
                    "fraction_of_kept_invalid": float(count / total_kept_invalid),
                    "total_kept_invalid": int(invalid_category_counts[band_name]["total_kept_invalid"]),
                    "total_invalid_in_category": int(invalid_category_totals[band_name][category]),
                }
            )

    score_samples = pd.concat(
        [
            score_samples_by_band[band_name].assign(split=assignments.split, band=band_name)
            for band_name in BAND_NAMES
        ],
        ignore_index=True,
    )
    if "_sample_key" in score_samples.columns:
        score_samples = score_samples.drop(columns="_sample_key")

    return BandHitFilterDiagnostics(
        split=assignments.split,
        score_samples=score_samples,
        threshold_sweep=pd.DataFrame(threshold_rows),
        region_summary=pd.DataFrame(region_rows),
        invalid_category_summary=pd.DataFrame(invalid_category_rows),
    )


def _write_band_score_summary_csv(out_path: Path, score_samples: pd.DataFrame) -> None:
    rows: list[dict[str, float | int | str]] = []
    for (band_name, truth_label), group in score_samples.groupby(["band", "truth_label"], sort=False):
        values = group["score"].to_numpy(dtype=np.float64)
        stats = _feature_stats(values)
        rows.append(
            {
                "split": str(group["split"].iloc[0]),
                "band": band_name,
                "truth_label": truth_label,
                "sample_size": int(len(group)),
                **stats,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _plot_band_score_diagnostics(
    diagnostics: BandHitFilterDiagnostics,
    *,
    output_base: Path,
    current_threshold: float,
) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.28)
    axes = axes.ravel()

    score_samples = diagnostics.score_samples
    num_events_by_band = {
        band_name: int(
            diagnostics.threshold_sweep.loc[diagnostics.threshold_sweep["band"] == band_name, "num_events"].iloc[0]
        )
        for band_name in BAND_NAMES
    }
    total_events_all_bands = int(sum(num_events_by_band.values()))
    threshold_zero = diagnostics.threshold_sweep.loc[np.isclose(diagnostics.threshold_sweep["threshold"], 0.0)].copy()
    if threshold_zero.empty:
        raise RuntimeError("Threshold sweep must include threshold=0.0 to normalize score-distribution panels.")
    bins = np.linspace(0.0, 1.0, 61)
    for idx, truth_label in enumerate(("valid", "invalid")):
        ax = axes[idx]
        for band_name in BAND_NAMES:
            values = score_samples.loc[
                (score_samples["band"] == band_name) & (score_samples["truth_label"] == truth_label),
                "score",
            ].to_numpy(dtype=np.float64)
            total_population = float(
                threshold_zero.loc[threshold_zero["band"] == band_name, "mean_kept_true_hits_per_event"].iloc[0]
                if truth_label == "valid"
                else threshold_zero.loc[threshold_zero["band"] == band_name, "mean_kept_invalid_hits_per_event"].iloc[0]
            ) * float(num_events_by_band[band_name])
            ax.hist(
                values,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(values),
                    sampled_population_size=len(values),
                    total_population=int(round(total_population)),
                    num_events=num_events_by_band[band_name],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5,
                color=BAND_COLORS[band_name],
                label=band_name.replace("_", " "),
            )
        ax.axvline(current_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{truth_label.title()}-hit score distribution")
        ax.set_xlabel("Hit-filter score")
        ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    for ax, band_name in zip(axes[2:4], BAND_NAMES, strict=True):
        for truth_label, color in (("valid", "#1b9e77"), ("invalid", "#d95f02")):
            values = score_samples.loc[
                (score_samples["band"] == band_name) & (score_samples["truth_label"] == truth_label),
                "score",
            ].to_numpy(dtype=np.float64)
            total_population = float(
                threshold_zero.loc[threshold_zero["band"] == band_name, "mean_kept_true_hits_per_event"].iloc[0]
                if truth_label == "valid"
                else threshold_zero.loc[threshold_zero["band"] == band_name, "mean_kept_invalid_hits_per_event"].iloc[0]
            ) * float(num_events_by_band[band_name])
            ax.hist(
                values,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(values),
                    sampled_population_size=len(values),
                    total_population=int(round(total_population)),
                    num_events=num_events_by_band[band_name],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5,
                color=color,
                label=truth_label,
            )
        ax.axvline(current_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{band_name.replace('_', ' ')}: valid vs invalid scores")
        ax.set_xlabel("Hit-filter score")
        ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    category_df = diagnostics.invalid_category_summary.copy()
    category_order = ["near_true_same_layer", "nonreco_truth_far", "noise_far"]
    category_labels = {
        "near_true_same_layer": "Near true hit",
        "nonreco_truth_far": "Nonreco truth",
        "noise_far": "Noise",
    }
    x = np.arange(len(BAND_NAMES), dtype=np.float64)
    bottom = np.zeros(len(BAND_NAMES), dtype=np.float64)
    category_colors = {
        "near_true_same_layer": "#4c78a8",
        "nonreco_truth_far": "#f58518",
        "noise_far": "#54a24b",
    }
    for category in category_order:
        heights = np.array(
            [
                float(
                    category_df.loc[
                        (category_df["band"] == band_name) & (category_df["category"] == category),
                        "fraction_of_kept_invalid",
                    ].iloc[0]
                )
                for band_name in BAND_NAMES
            ],
            dtype=np.float64,
        )
        axes[4].bar(x, heights, bottom=bottom, color=category_colors[category], label=category_labels[category], alpha=0.9)
        bottom += heights
    axes[4].set_xticks(x, [band_name.replace("_", " ") for band_name in BAND_NAMES])
    axes[4].set_ylim(0.0, 1.0)
    axes[4].set_ylabel("Fraction of kept invalid hits")
    axes[4].set_title("Invalid-hit composition at current threshold")
    axes[4].grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="y")
    axes[4].set_axisbelow(True)

    invalid_only = score_samples[score_samples["truth_label"] == "invalid"].copy()
    for category in category_order:
        category_values: list[np.ndarray] = []
        category_weights: list[np.ndarray] = []
        for band_name in BAND_NAMES:
            values = invalid_only.loc[
                (invalid_only["band"] == band_name) & (invalid_only["invalid_category"] == category),
                "score",
            ].to_numpy(dtype=np.float64)
            total_population = int(
                diagnostics.invalid_category_summary.loc[
                    (diagnostics.invalid_category_summary["band"] == band_name)
                    & (diagnostics.invalid_category_summary["category"] == category),
                    "total_invalid_in_category",
                ].iloc[0]
            )
            weights = _sampled_population_hist_weights(
                len(values),
                sampled_population_size=len(values),
                total_population=total_population,
                num_events=total_events_all_bands,
            )
            if len(values) > 0:
                category_values.append(values)
                if weights is None:
                    category_weights.append(np.ones(len(values), dtype=np.float64))
                else:
                    category_weights.append(weights)
        plot_values = np.concatenate(category_values) if category_values else np.array([], dtype=np.float64)
        plot_weights = None if _hist_density() else (np.concatenate(category_weights) if category_weights else None)
        axes[5].hist(
            plot_values,
            bins=bins,
            weights=plot_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=category_colors[category],
            label=category_labels[category],
        )
    axes[5].axvline(current_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[5].set_title("Invalid-hit score distributions by category")
    axes[5].set_xlabel("Hit-filter score")
    axes[5].set_ylabel(_hist_ylabel(allow_count_per_event=True))
    axes[5].grid(alpha=0.10, linestyle="--", linewidth=0.45)
    axes[5].set_axisbelow(True)

    axes[0].legend(frameon=False, loc="best")
    axes[2].legend(frameon=False, loc="best")
    axes[4].legend(frameon=False, loc="upper right")
    axes[5].legend(frameon=False, loc="best")
    fig.suptitle(f"{diagnostics.split}: pre-threshold score diagnostics by band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_band_threshold_sweep(diagnostics: BandHitFilterDiagnostics, output_base: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.6), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.28)
    axes = axes.ravel()

    threshold_df = diagnostics.threshold_sweep
    panels = [
        ("mean_kept_invalid_hits_per_event", "Kept invalid hits / event"),
        ("mean_contamination_fraction", "Contamination fraction"),
        ("mean_kept_true_hits_per_event", "Kept true hits / event"),
        ("mean_true_hit_recall", "True-hit recall"),
        ("mean_reconstructable_tracks_per_event", "Reconstructable tracks / event"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels, strict=False):
        for band_name in BAND_NAMES:
            band_df = threshold_df[threshold_df["band"] == band_name]
            ax.plot(
                band_df["threshold"],
                band_df[metric],
                color=BAND_COLORS[band_name],
                linewidth=1.8,
                label=band_name.replace("_", " "),
            )
        ax.set_xlabel("Hit-filter threshold")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    axes[5].axis("off")
    axes[5].legend(
        [axes[0].lines[0], axes[0].lines[1]],
        [BAND_NAMES[0].replace("_", " "), BAND_NAMES[1].replace("_", " ")],
        loc="center",
        frameon=False,
    )
    fig.suptitle(f"{diagnostics.split}: threshold-response by post-filter band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_band_region_conditioned_diagnostics(
    diagnostics: BandHitFilterDiagnostics,
    *,
    output_base: Path,
    top_region_keys: int,
) -> None:
    plt = _get_pyplot()
    region_df = diagnostics.region_summary.copy()
    if region_df.empty:
        return

    pivot_contamination = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="kept_contamination_fraction",
        aggfunc="first",
    )
    pivot_score_gap = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="score_gap_pre",
        aggfunc="first",
    )
    pivot_near_true = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="kept_invalid_near_true_fraction",
        aggfunc="first",
    )
    for pivot in (pivot_contamination, pivot_score_gap, pivot_near_true):
        for band_name in BAND_NAMES:
            if band_name not in pivot.columns:
                pivot[band_name] = np.nan

    ranking = (
        pivot_contamination[BAND_NAMES[0]].fillna(0.0) - pivot_contamination[BAND_NAMES[1]].fillna(0.0)
    ).abs().sort_values(ascending=False)
    selected_keys = list(ranking.head(max(1, top_region_keys)).index)
    labels = [f"V{int(volume_id)}/L{int(layer_id)}" for volume_id, layer_id in selected_keys]
    y = np.arange(len(selected_keys), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, max(5.6, 0.42 * len(selected_keys) + 2.4)), constrained_layout=False)
    fig.subplots_adjust(top=0.88, wspace=0.38)
    panel_specs = [
        (pivot_contamination, "kept contamination", "Kept contamination fraction"),
        (pivot_score_gap, "score gap", "Mean valid - invalid score"),
        (pivot_near_true, "near-true invalid fraction", "Fraction of kept invalid near true"),
    ]
    for ax, (pivot, title, xlabel) in zip(axes, panel_specs, strict=True):
        lower_vals = np.array([pivot.loc[key, BAND_NAMES[0]] for key in selected_keys], dtype=np.float64)
        upper_vals = np.array([pivot.loc[key, BAND_NAMES[1]] for key in selected_keys], dtype=np.float64)
        ax.barh(y + 0.18, lower_vals, height=0.34, color=BAND_COLORS[BAND_NAMES[0]], label=BAND_NAMES[0].replace("_", " "), alpha=0.85)
        ax.barh(y - 0.18, upper_vals, height=0.34, color=BAND_COLORS[BAND_NAMES[1]], label=BAND_NAMES[1].replace("_", " "), alpha=0.85)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Top regions by {title}")
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="x")
        ax.set_axisbelow(True)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{diagnostics.split}: detector-region conditioned band diagnostics", fontsize=13, y=0.965)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_explainer_feature_label(feature: str) -> str:
    custom_labels = {
        "hits_on_reconstructable_tracks_after_filter": "Kept hits on reconstructable tracks",
        "hits_off_reconstructable_tracks_after_filter": "Kept hits off reconstructable tracks",
        "filtered_hit_purity": "Filtered-hit purity",
        "filtered_hit_contamination": "Filtered-hit contamination",
        "off_track_hit_fraction_on_truth_particles": "Off-track kept hits on truth particles",
        "off_track_hit_fraction_noise": "Off-track kept hits on noise",
        "track_pt_mean_pre": "Mean track pT before filtering",
        "track_pt_median_pre": "Median track pT before filtering",
        "track_pt_p10_pre": "Track pT p10 before filtering",
        "track_pt_p90_pre": "Track pT p90 before filtering",
        "track_pt_sum_pre": "Scalar sum track pT before filtering",
        "fraction_tracks_pt_lt_2_pre": "Fraction of tracks with pT < 2 GeV",
        "fraction_tracks_pt_lt_5_pre": "Fraction of tracks with pT < 5 GeV",
        "track_abs_eta_mean_pre": "Mean |eta| before filtering",
        "fraction_tracks_abs_eta_gt_2_pre": "Fraction of tracks with |eta| > 2",
        "num_occupied_volume_layers_true": "Occupied volume-layers with true hits",
        "mean_true_hits_per_occupied_volume_layer": "Mean true hits per occupied volume-layer",
        "max_true_hits_single_volume_layer": "Max true hits in one volume-layer",
        "mean_true_hit_phi_nn_gap_same_layer": "Mean true-hit phi NN gap in layer",
        "p10_true_hit_phi_nn_gap_same_layer": "True-hit phi NN gap p10 in layer",
        "fraction_true_hits_with_phi_nn_lt_0p02_same_layer": "True hits with close same-layer phi neighbor",
        "mean_true_hit_rphiz_nn_same_layer": "Mean true-hit same-layer rphi/z NN distance",
        "p10_true_hit_rphiz_nn_same_layer": "True-hit same-layer rphi/z NN p10",
        "fraction_true_hits_with_close_rphiz_nn_same_layer": "True hits with close same-layer rphi/z NN",
        "mean_true_hit_neighbors_within_crowding_distance_same_layer": "Mean same-layer neighbors around true hits",
        "fraction_true_hits_with_5plus_neighbors_same_layer": "True hits with >=5 same-layer neighbors",
        "mean_true_track_phi_nn_gap": "Mean truth-track phi NN gap",
        "p10_true_track_phi_nn_gap": "Truth-track phi NN gap p10",
        "fraction_true_tracks_with_phi_nn_lt_0p05": "Truth tracks with close phi neighbor",
        "mean_true_track_delta_r_nn": "Mean truth-track deltaR NN distance",
        "p10_true_track_delta_r_nn": "Truth-track deltaR NN p10",
        "fraction_true_tracks_with_delta_r_nn_lt_0p10": "Truth tracks with close deltaR neighbor",
        "mean_pre_hits_per_surviving_track": "Mean pre hits per surviving track",
        "mean_post_hits_per_reconstructable_track": "Mean post hits per surviving track",
        "mean_removed_true_hits_per_surviving_track": "Mean removed true hits per surviving track",
        "mean_track_hit_retention": "Mean track hit retention",
        "mean_hit_shortfall_surviving_track": "Mean surviving-track hit shortfall",
        "event_hit_retention_on_surviving_tracks": "Event hit retention on surviving tracks",
        "fraction_tracks_with_3_post_hits": "Tracks with 3 post hits",
        "fraction_tracks_with_4_post_hits": "Tracks with 4 post hits",
        "fraction_tracks_with_5_post_hits": "Tracks with 5 post hits",
        "fraction_tracks_with_6_post_hits": "Tracks with 6 post hits",
        "fraction_tracks_with_7_post_hits": "Tracks with 7 post hits",
        "fraction_tracks_with_8plus_post_hits": "Tracks with >=8 post hits",
        "fraction_tracks_with_4_or_less_post_hits": "Tracks with <=4 post hits",
        "fraction_surviving_tracks_lost_0_hits": "Surviving tracks losing 0 hits",
        "fraction_surviving_tracks_lost_1_hit": "Surviving tracks losing 1 hit",
        "fraction_surviving_tracks_lost_2plus_hits": "Surviving tracks losing >=2 hits",
        "mean_invalid_hit_distance_to_true_same_layer": "Mean invalid-hit distance to true hit in layer",
        "kept_invalid_hits_near_true_same_layer": "Kept invalid hits near true hits",
        "kept_invalid_hits_on_nonreco_truth_far": "Kept invalid hits on nonreco truth far from true",
        "kept_invalid_hits_noise_far": "Kept invalid noise hits far from true",
        "fraction_invalid_hits_near_true_same_layer": "Invalid hits near true hits",
        "fraction_invalid_hits_on_nonreco_truth_far": "Invalid hits on nonreco truth far from true",
        "fraction_invalid_hits_noise_far": "Invalid noise hits far from true",
        "score_mean_all_pre": "Mean pre-filter score",
        "score_std_all_pre": "Pre-filter score std",
        "score_mean_valid_pre": "Mean score on valid hits",
        "score_mean_invalid_pre": "Mean score on invalid hits",
        "score_mean_valid_minus_invalid_pre": "Score gap: valid - invalid",
        "score_fraction_valid_above_threshold_pre": "Valid-hit score > threshold",
        "score_fraction_invalid_above_threshold_pre": "Invalid-hit score > threshold",
        "score_fraction_valid_near_threshold_pre": "Valid-hit scores near threshold",
        "score_fraction_invalid_near_threshold_pre": "Invalid-hit scores near threshold",
        "score_margin_fraction_pre": "Scores near threshold",
    }
    if feature in custom_labels:
        return custom_labels[feature]
    if feature.startswith("hit_fraction_volume_"):
        suffix = feature.removeprefix("hit_fraction_volume_")
        return f"Filtered-hit fraction V/L {suffix.replace('_layer_', '/')}" if "_layer_" in suffix else f"Filtered-hit fraction V{suffix}"
    if feature.startswith("invalid_hit_fraction_volume_"):
        suffix = feature.removeprefix("invalid_hit_fraction_volume_")
        return f"Off-track hit fraction V/L {suffix.replace('_layer_', '/')}" if "_layer_" in suffix else f"Off-track hit fraction V{suffix}"
    if feature.startswith("retention_volume_"):
        suffix = feature.removeprefix("retention_volume_")
        return f"Retention V/L {suffix.replace('_layer_', '/')}" if "_layer_" in suffix else f"Retention V{suffix}"
    return feature.replace("_", " ")


def _select_explainer_feature_columns(diagnostics: pd.DataFrame) -> list[str]:
    feature_columns: list[str] = []
    for column in diagnostics.columns:
        if column in EXPLAINER_EXCLUDED_COLUMNS:
            continue
        if diagnostics[column].dtype.kind not in {"b", "i", "u", "f"}:
            continue
        values = diagnostics[column].to_numpy(dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        if np.nanstd(finite_values) < 1e-12:
            continue
        feature_columns.append(column)
    return feature_columns


def _hit_bin_codes(hits_after_filter: np.ndarray, num_bins: int) -> np.ndarray:
    n_unique = int(np.unique(hits_after_filter).size)
    if n_unique <= 1:
        return np.zeros(len(hits_after_filter), dtype=np.int32)
    n_bins = max(2, min(num_bins, n_unique))
    return np.asarray(pd.qcut(hits_after_filter, q=n_bins, labels=False, duplicates="drop"), dtype=np.int32)


def _residualize_features_by_hit_count(
    diagnostics: pd.DataFrame,
    feature_columns: list[str],
    num_hit_bins: int,
) -> pd.DataFrame:
    bin_codes = _hit_bin_codes(diagnostics["hits_after_filter"].to_numpy(dtype=np.float64), num_bins=num_hit_bins)
    residualized: dict[str, np.ndarray] = {}
    for column in feature_columns:
        values = diagnostics[column].to_numpy(dtype=np.float64)
        transformed = np.zeros(len(values), dtype=np.float64)
        for bin_code in np.unique(bin_codes):
            mask = bin_codes == bin_code
            finite_mask = mask & np.isfinite(values)
            if not np.any(finite_mask):
                continue
            bin_values = values[finite_mask]
            mean = float(np.mean(bin_values))
            std = float(np.std(bin_values))
            if std < 1e-12:
                transformed[finite_mask] = 0.0
            else:
                transformed[finite_mask] = (bin_values - mean) / std
        residualized[column] = transformed
    residualized_df = pd.DataFrame(residualized, index=diagnostics.index)
    residualized_df.insert(0, "hit_bin_code", bin_codes)
    return residualized_df


def _fit_band_explainer(
    diagnostics: pd.DataFrame,
    split: str,
    num_hit_bins: int,
    test_fraction: float,
) -> BandExplainerResults:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for the band explainer. Install it in the active environment and rerun."
        ) from exc

    feature_columns = _select_explainer_feature_columns(diagnostics)
    if not feature_columns:
        raise RuntimeError(f"Cannot fit band explainer for split '{split}': no usable numeric feature columns.")
    residualized = _residualize_features_by_hit_count(diagnostics, feature_columns=feature_columns, num_hit_bins=num_hit_bins)

    core_mask = diagnostics["position_group"].isin(["lower_core", "upper_core"]).to_numpy(dtype=bool)
    y_all = (diagnostics["band"] == "upper_band").to_numpy(dtype=np.int32)
    X_all = residualized.loc[:, feature_columns].to_numpy(dtype=np.float64)
    X_core = X_all[core_mask]
    y_core = y_all[core_mask]

    if len(np.unique(y_core)) < 2:
        raise RuntimeError(f"Cannot fit band explainer for split '{split}': need both lower_core and upper_core events.")
    if np.min(np.bincount(y_core)) < 2:
        raise RuntimeError(f"Cannot fit band explainer for split '{split}': each core class needs at least two events.")

    if X_core.shape[0] < 10:
        raise RuntimeError(f"Cannot fit band explainer for split '{split}': not enough core events ({X_core.shape[0]}).")

    requested_test_size = int(round(test_fraction * X_core.shape[0])) if test_fraction < 1.0 else int(test_fraction)
    requested_test_size = max(2, requested_test_size)
    requested_test_size = min(requested_test_size, X_core.shape[0] - 2)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_core,
        y_core,
        np.flatnonzero(core_mask),
        test_size=requested_test_size,
        random_state=12345,
        stratify=y_core,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    all_prob = model.predict_proba(X_all)[:, 1]
    coef = model.named_steps["logisticregression"].coef_[0].astype(np.float64)

    core_df = residualized.loc[core_mask, feature_columns]
    lower_mask = diagnostics.loc[core_mask, "band"] == "lower_band"
    upper_mask = diagnostics.loc[core_mask, "band"] == "upper_band"
    mean_lower = core_df.loc[lower_mask, :].mean(axis=0).to_numpy(dtype=np.float64)
    mean_upper = core_df.loc[upper_mask, :].mean(axis=0).to_numpy(dtype=np.float64)
    delta = mean_upper - mean_lower

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "label": [_format_explainer_feature_label(column) for column in feature_columns],
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "mean_lower_core_z": mean_lower,
            "mean_upper_core_z": mean_upper,
            "upper_minus_lower_z": delta,
            "abs_upper_minus_lower_z": np.abs(delta),
        }
    ).sort_values("abs_coefficient", ascending=False, ignore_index=True)

    predictions = diagnostics.loc[
        :,
        ["split", "event_index", "sample_id", "event_name", "band", "position_group", "band_position", "upper_band_posterior"],
    ].copy()
    predictions["explainer_upper_band_probability"] = all_prob
    predictions["explainer_label"] = np.where(all_prob >= 0.5, "upper_band", "lower_band")
    predictions["is_core_event"] = core_mask
    predictions["is_test_event"] = False
    predictions.loc[test_idx, "is_test_event"] = True

    summary = {
        "num_features": float(len(feature_columns)),
        "num_events": float(len(diagnostics)),
        "num_core_events": float(np.sum(core_mask)),
        "train_roc_auc": float(roc_auc_score(y_train, train_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_average_precision": float(average_precision_score(y_test, test_prob)),
        "test_accuracy": float(accuracy_score(y_test, test_prob >= 0.5)),
        "core_fullsample_roc_auc": float(roc_auc_score(y_core, all_prob[core_mask])),
    }
    return BandExplainerResults(
        split=split,
        feature_columns=feature_columns,
        residualized_features=residualized,
        predictions=predictions,
        coefficients=coefficients,
        summary=summary,
    )


def _write_band_explainer_summary_csv(out_path: Path, results: BandExplainerResults) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["split", results.split])
        for metric, value in results.summary.items():
            writer.writerow([metric, value])


def _plot_band_explainer(
    diagnostics: pd.DataFrame,
    results: BandExplainerResults,
    output_base: Path,
    top_features: int,
) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.36, wspace=0.30)
    ax_coef, ax_shift, ax_prob, ax_order = axes.ravel()

    coef_df = results.coefficients.nlargest(top_features, "abs_coefficient").iloc[::-1]
    coef_colors = [BAND_COLORS["upper_band"] if value > 0 else BAND_COLORS["lower_band"] for value in coef_df["coefficient"]]
    ax_coef.barh(coef_df["label"], coef_df["coefficient"], color=coef_colors, alpha=0.85)
    ax_coef.axvline(0.0, color="#666666", linewidth=0.8)
    ax_coef.set_title("Top logistic coefficients")
    ax_coef.set_xlabel("Coefficient sign: upper vs lower band")
    ax_coef.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="x")
    ax_coef.set_axisbelow(True)

    shift_df = results.coefficients.nlargest(top_features, "abs_upper_minus_lower_z").iloc[::-1]
    shift_colors = [BAND_COLORS["upper_band"] if value > 0 else BAND_COLORS["lower_band"] for value in shift_df["upper_minus_lower_z"]]
    ax_shift.barh(shift_df["label"], shift_df["upper_minus_lower_z"], color=shift_colors, alpha=0.85)
    ax_shift.axvline(0.0, color="#666666", linewidth=0.8)
    ax_shift.set_title("Top hit-conditioned feature shifts")
    ax_shift.set_xlabel("Upper-core minus lower-core z-shift")
    ax_shift.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="x")
    ax_shift.set_axisbelow(True)

    group_colors = {
        "lower_core": BAND_COLORS["lower_band"],
        "between_bands": "#777777",
        "upper_core": BAND_COLORS["upper_band"],
    }
    for group_name, color in group_colors.items():
        mask = diagnostics["position_group"] == group_name
        ax_prob.scatter(
            diagnostics.loc[mask, "band_position"],
            results.predictions.loc[mask, "explainer_upper_band_probability"],
            s=10.0,
            alpha=0.55,
            color=color,
            edgecolors="none",
            rasterized=True,
            label=group_name.replace("_", " "),
        )
    ax_prob.set_title("Explainer probability vs band position")
    ax_prob.set_xlabel("Position between fitted bands")
    ax_prob.set_ylabel("Explainer P(upper band)")
    ax_prob.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    ax_prob.set_axisbelow(True)
    ax_prob.legend(frameon=False, loc="best", title=f"test ROC AUC = {results.summary['test_roc_auc']:.3f}")

    for group_name, color in group_colors.items():
        mask = diagnostics["position_group"] == group_name
        ax_order.scatter(
            diagnostics.loc[mask, "sample_id"],
            diagnostics.loc[mask, "band_position"],
            s=10.0,
            alpha=0.45,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
    ax_order.set_title("Band position vs sample id")
    ax_order.set_xlabel("Sample id")
    ax_order.set_ylabel("Position between fitted bands")
    ax_order.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    ax_order.set_axisbelow(True)

    fig.suptitle(f"{results.split}: event-level band explainer", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _priority_sample(existing: pd.DataFrame, new_rows: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    if new_rows.empty:
        return existing

    new_rows = new_rows.copy()
    new_rows["_sample_key"] = rng.random(len(new_rows))
    if existing.empty:
        combined = new_rows
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)

    if max_rows > 0 and len(combined) > max_rows:
        combined = combined.nsmallest(max_rows, "_sample_key").reset_index(drop=True)

    return combined


def _collect_band_truth_trackloss_feature_samples(
    cfg: dict,
    assignments: BandAssignments,
    *,
    hit_selection: str,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    rng_seed: int,
) -> tuple[TruthBandSamples, BandTrackLossSamples, BandFeatureSamples]:
    if assignments.hit_eval_path is None:
        raise RuntimeError(
            f"Combined band sample analysis requires hit filtering, but split '{assignments.split}' has no hit_eval_path."
        )

    dataset_post, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=True)
    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    rng_truth = np.random.default_rng(rng_seed + 1000)
    rng_track_loss = np.random.default_rng(rng_seed + 5000)
    rng_feature = np.random.default_rng(rng_seed)

    truth_hit_samples = {band_name: pd.DataFrame(columns=[*HIT_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    truth_track_samples = {band_name: pd.DataFrame(columns=[*TRUE_TRACK_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    truth_event_metrics = {
        band_name: {
            "event_true_tracks": [],
            "event_true_hits": [],
            "event_true_hits_per_true_track": [],
        }
        for band_name in BAND_NAMES
    }
    truth_totals = {
        band_name: {
            "num_events": 0,
            "num_true_hits": 0,
            "num_true_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    track_loss_samples = {
        band_name: pd.DataFrame(
            columns=[
                "pt",
                "eta",
                "phi",
                "num_hits_pre",
                "num_hits_post",
                "num_hits_lost",
                "hit_retention",
                "survived_filter",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    track_loss_event_metrics = {
        band_name: {
            "event_truth_tracks": [],
            "event_surviving_tracks": [],
            "event_lost_tracks": [],
            "event_lost_track_fraction": [],
            "event_mean_hits_lost_per_truth_track": [],
            "event_mean_hit_retention_per_truth_track": [],
        }
        for band_name in BAND_NAMES
    }
    track_loss_totals = {
        band_name: {
            "num_events": 0,
            "num_truth_tracks": 0,
            "num_surviving_tracks": 0,
            "num_lost_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    feature_hit_samples = {band_name: pd.DataFrame(columns=[*HIT_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    feature_particle_samples = {band_name: pd.DataFrame(columns=[*PARTICLE_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    feature_event_hits_per_track = {band_name: [] for band_name in BAND_NAMES}
    feature_totals = {
        band_name: {
            "num_events": 0,
            "num_filtered_hits": 0,
            "num_selected_hits": 0,
            "num_reconstructable_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    print(
        f"[{assignments.split}] collecting truth/track-loss/feature samples in one pass "
        f"(hit_selection={hit_selection}, max_hit_samples={max_hit_samples_per_band}, "
        f"max_particle_samples={max_particle_samples_per_band})"
    )
    for event_idx, band_code in enumerate(assignments.band_codes):
        band_name = BAND_NAMES[int(band_code)]
        hits_post, particles_post = dataset_post.load_event(event_idx)
        hits_truth, particles_truth = dataset_truth.load_event(event_idx)

        true_hits = hits_truth[hits_truth["on_valid_particle"]]
        valid_hits_post = hits_post[hits_post["on_valid_particle"]]
        truth_particle_ids = particles_truth["particle_id"]
        true_hit_counts_series = true_hits.groupby("particle_id").size()
        valid_post_hit_counts_series = valid_hits_post.groupby("particle_id").size()
        truth_hit_counts_series = hits_truth.groupby("particle_id").size()
        true_hit_counts = (
            true_hit_counts_series
            .reindex(truth_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )

        truth_hit_frame = true_hits.loc[:, list(HIT_FEATURE_COLUMNS)]
        truth_track_frame = particles_truth.loc[:, ["pt", "eta", "phi"]].copy()
        truth_track_frame["num_true_hits"] = true_hit_counts

        truth_hit_samples[band_name] = _priority_sample(
            truth_hit_samples[band_name],
            truth_hit_frame,
            max_rows=max_hit_samples_per_band,
            rng=rng_truth,
        )
        truth_track_samples[band_name] = _priority_sample(
            truth_track_samples[band_name],
            truth_track_frame,
            max_rows=max_particle_samples_per_band,
            rng=rng_truth,
        )

        truth_totals[band_name]["num_events"] += 1
        truth_totals[band_name]["num_true_hits"] += int(len(true_hits))
        truth_totals[band_name]["num_true_tracks"] += int(len(particles_truth))
        truth_event_metrics[band_name]["event_true_tracks"].append(int(len(particles_truth)))
        truth_event_metrics[band_name]["event_true_hits"].append(int(len(true_hits)))
        truth_event_metrics[band_name]["event_true_hits_per_true_track"].append(
            float(len(true_hits)) / max(float(len(particles_truth)), 1.0)
        )

        post_particle_ids = particles_post["particle_id"]
        pre_hit_counts_truth = (
            true_hit_counts_series
            .reindex(truth_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        post_hit_counts_truth = (
            valid_post_hit_counts_series
            .reindex(truth_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        num_hits_lost = pre_hit_counts_truth - post_hit_counts_truth
        survived_filter = truth_particle_ids.isin(post_particle_ids).to_numpy(dtype=np.int32)
        track_hit_retention = np.divide(
            post_hit_counts_truth,
            pre_hit_counts_truth,
            out=np.zeros_like(post_hit_counts_truth, dtype=np.float64),
            where=pre_hit_counts_truth > 0,
        )

        track_loss_frame = particles_truth.loc[:, ["pt", "eta", "phi"]].copy()
        track_loss_frame["num_hits_pre"] = pre_hit_counts_truth
        track_loss_frame["num_hits_post"] = post_hit_counts_truth
        track_loss_frame["num_hits_lost"] = num_hits_lost
        track_loss_frame["hit_retention"] = track_hit_retention
        track_loss_frame["survived_filter"] = survived_filter

        track_loss_samples[band_name] = _priority_sample(
            track_loss_samples[band_name],
            track_loss_frame,
            max_rows=max_particle_samples_per_band,
            rng=rng_track_loss,
        )

        num_truth_tracks = int(len(particles_truth))
        num_surviving_tracks = int(np.sum(survived_filter))
        num_lost_tracks = int(num_truth_tracks - num_surviving_tracks)
        track_loss_totals[band_name]["num_events"] += 1
        track_loss_totals[band_name]["num_truth_tracks"] += num_truth_tracks
        track_loss_totals[band_name]["num_surviving_tracks"] += num_surviving_tracks
        track_loss_totals[band_name]["num_lost_tracks"] += num_lost_tracks
        track_loss_event_metrics[band_name]["event_truth_tracks"].append(num_truth_tracks)
        track_loss_event_metrics[band_name]["event_surviving_tracks"].append(num_surviving_tracks)
        track_loss_event_metrics[band_name]["event_lost_tracks"].append(num_lost_tracks)
        track_loss_event_metrics[band_name]["event_lost_track_fraction"].append(float(num_lost_tracks / max(num_truth_tracks, 1)))
        track_loss_event_metrics[band_name]["event_mean_hits_lost_per_truth_track"].append(
            float(np.mean(num_hits_lost.astype(np.float64))) if num_truth_tracks > 0 else np.nan
        )
        track_loss_event_metrics[band_name]["event_mean_hit_retention_per_truth_track"].append(
            float(np.mean(track_hit_retention.astype(np.float64))) if num_truth_tracks > 0 else np.nan
        )

        if hit_selection == "all":
            selected_hits = hits_post
        elif hit_selection == "valid-track":
            selected_hits = valid_hits_post
        else:
            raise ValueError(f"Unknown hit_selection={hit_selection!r}")

        post_hit_counts_post = (
            valid_post_hit_counts_series
            .reindex(post_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        pre_hit_counts_post = (
            truth_hit_counts_series
            .reindex(post_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        feature_hit_retention = np.divide(
            post_hit_counts_post,
            pre_hit_counts_post,
            out=np.zeros_like(post_hit_counts_post, dtype=np.float64),
            where=pre_hit_counts_post > 0,
        )

        feature_hit_frame = selected_hits.loc[:, list(HIT_FEATURE_COLUMNS)]
        feature_particle_frame = particles_post.loc[:, ["pt", "eta", "phi"]].copy()
        feature_particle_frame["num_hits_post"] = post_hit_counts_post
        feature_particle_frame["num_hits_pre"] = pre_hit_counts_post
        feature_particle_frame["hit_retention"] = feature_hit_retention

        feature_hit_samples[band_name] = _priority_sample(
            feature_hit_samples[band_name],
            feature_hit_frame,
            max_rows=max_hit_samples_per_band,
            rng=rng_feature,
        )
        feature_particle_samples[band_name] = _priority_sample(
            feature_particle_samples[band_name],
            feature_particle_frame,
            max_rows=max_particle_samples_per_band,
            rng=rng_feature,
        )

        feature_totals[band_name]["num_events"] += 1
        feature_totals[band_name]["num_filtered_hits"] += int(len(hits_post))
        feature_totals[band_name]["num_selected_hits"] += int(len(selected_hits))
        feature_totals[band_name]["num_reconstructable_tracks"] += int(len(particles_post))
        feature_event_hits_per_track[band_name].append(float(len(hits_post)) / max(float(len(particles_post)), 1.0))

        if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
            print(f"[{assignments.split}] combined truth/track-loss/feature events processed {event_idx + 1}/{len(assignments.band_codes)}")

    truth_samples = TruthBandSamples(
        split=assignments.split,
        hit_samples=truth_hit_samples,
        track_samples=truth_track_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in truth_event_metrics.items()
        },
        totals=truth_totals,
    )
    track_loss_result = BandTrackLossSamples(
        split=assignments.split,
        track_samples=track_loss_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in track_loss_event_metrics.items()
        },
        totals=track_loss_totals,
    )
    feature_samples = BandFeatureSamples(
        split=assignments.split,
        hit_selection=hit_selection,
        hit_samples=feature_hit_samples,
        particle_samples=feature_particle_samples,
        event_hits_per_track={k: np.asarray(v, dtype=np.float64) for k, v in feature_event_hits_per_track.items()},
        totals=feature_totals,
    )
    return truth_samples, track_loss_result, feature_samples


def _collect_truth_band_samples(
    cfg: dict,
    assignments: BandAssignments,
    max_hit_samples_per_band: int,
    max_track_samples_per_band: int,
    rng_seed: int,
) -> TruthBandSamples:
    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    rng = np.random.default_rng(rng_seed + 1000)

    hit_samples = {band_name: pd.DataFrame(columns=[*HIT_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    track_samples = {band_name: pd.DataFrame(columns=[*TRUE_TRACK_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    event_metrics = {
        band_name: {
            "event_true_tracks": [],
            "event_true_hits": [],
            "event_true_hits_per_true_track": [],
        }
        for band_name in BAND_NAMES
    }
    totals = {
        band_name: {
            "num_events": 0,
            "num_true_hits": 0,
            "num_true_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    print(
        f"[{assignments.split}] collecting truth-band samples "
        f"(max_hit_samples={max_hit_samples_per_band}, max_track_samples={max_track_samples_per_band})"
    )
    for event_idx, band_code in enumerate(assignments.band_codes):
        band_name = BAND_NAMES[int(band_code)]
        hits_truth, particles_truth = dataset_truth.load_event(event_idx)
        true_hits = hits_truth[hits_truth["on_valid_particle"]]

        particle_ids = particles_truth["particle_id"]
        true_hit_counts = (
            true_hits.groupby("particle_id")
            .size()
            .reindex(particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )

        hit_frame = true_hits.loc[:, list(HIT_FEATURE_COLUMNS)]
        track_frame = particles_truth.loc[:, ["pt", "eta", "phi"]].copy()
        track_frame["num_true_hits"] = true_hit_counts

        hit_samples[band_name] = _priority_sample(
            hit_samples[band_name],
            hit_frame,
            max_rows=max_hit_samples_per_band,
            rng=rng,
        )
        track_samples[band_name] = _priority_sample(
            track_samples[band_name],
            track_frame,
            max_rows=max_track_samples_per_band,
            rng=rng,
        )

        totals[band_name]["num_events"] += 1
        totals[band_name]["num_true_hits"] += int(len(true_hits))
        totals[band_name]["num_true_tracks"] += int(len(particles_truth))
        event_metrics[band_name]["event_true_tracks"].append(int(len(particles_truth)))
        event_metrics[band_name]["event_true_hits"].append(int(len(true_hits)))
        event_metrics[band_name]["event_true_hits_per_true_track"].append(float(len(true_hits)) / max(float(len(particles_truth)), 1.0))

        if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
            print(f"[{assignments.split}] truth-band events processed {event_idx + 1}/{len(assignments.band_codes)}")

    return TruthBandSamples(
        split=assignments.split,
        hit_samples=hit_samples,
        track_samples=track_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in event_metrics.items()
        },
        totals=totals,
    )


def _collect_band_track_loss_samples(
    cfg: dict,
    assignments: BandAssignments,
    max_track_samples_per_band: int,
    rng_seed: int,
) -> BandTrackLossSamples:
    dataset_post, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=True)
    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    rng = np.random.default_rng(rng_seed + 5000)

    track_samples = {
        band_name: pd.DataFrame(
            columns=[
                "pt",
                "eta",
                "phi",
                "num_hits_pre",
                "num_hits_post",
                "num_hits_lost",
                "hit_retention",
                "survived_filter",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    event_metrics = {
        band_name: {
            "event_truth_tracks": [],
            "event_surviving_tracks": [],
            "event_lost_tracks": [],
            "event_lost_track_fraction": [],
            "event_mean_hits_lost_per_truth_track": [],
            "event_mean_hit_retention_per_truth_track": [],
        }
        for band_name in BAND_NAMES
    }
    totals = {
        band_name: {
            "num_events": 0,
            "num_truth_tracks": 0,
            "num_surviving_tracks": 0,
            "num_lost_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    print(
        f"[{assignments.split}] collecting reconstructable-track loss samples "
        f"(max_track_samples={max_track_samples_per_band})"
    )
    for event_idx, band_code in enumerate(assignments.band_codes):
        band_name = BAND_NAMES[int(band_code)]
        hits_post, particles_post = dataset_post.load_event(event_idx)
        hits_truth, particles_truth = dataset_truth.load_event(event_idx)

        truth_particle_ids = particles_truth["particle_id"]
        post_particle_ids = particles_post["particle_id"]
        pre_hit_counts = (
            hits_truth[hits_truth["on_valid_particle"]]
            .groupby("particle_id")
            .size()
            .reindex(truth_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        post_hit_counts = (
            hits_post[hits_post["on_valid_particle"]]
            .groupby("particle_id")
            .size()
            .reindex(truth_particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        num_hits_lost = pre_hit_counts - post_hit_counts
        survived_filter = truth_particle_ids.isin(post_particle_ids).to_numpy(dtype=np.int32)
        hit_retention = np.divide(
            post_hit_counts,
            pre_hit_counts,
            out=np.zeros_like(post_hit_counts, dtype=np.float64),
            where=pre_hit_counts > 0,
        )

        track_frame = particles_truth.loc[:, ["pt", "eta", "phi"]].copy()
        track_frame["num_hits_pre"] = pre_hit_counts
        track_frame["num_hits_post"] = post_hit_counts
        track_frame["num_hits_lost"] = num_hits_lost
        track_frame["hit_retention"] = hit_retention
        track_frame["survived_filter"] = survived_filter

        track_samples[band_name] = _priority_sample(
            track_samples[band_name],
            track_frame,
            max_rows=max_track_samples_per_band,
            rng=rng,
        )

        num_truth_tracks = int(len(particles_truth))
        num_surviving_tracks = int(np.sum(survived_filter))
        num_lost_tracks = int(num_truth_tracks - num_surviving_tracks)
        totals[band_name]["num_events"] += 1
        totals[band_name]["num_truth_tracks"] += num_truth_tracks
        totals[band_name]["num_surviving_tracks"] += num_surviving_tracks
        totals[band_name]["num_lost_tracks"] += num_lost_tracks
        event_metrics[band_name]["event_truth_tracks"].append(num_truth_tracks)
        event_metrics[band_name]["event_surviving_tracks"].append(num_surviving_tracks)
        event_metrics[band_name]["event_lost_tracks"].append(num_lost_tracks)
        event_metrics[band_name]["event_lost_track_fraction"].append(float(num_lost_tracks / max(num_truth_tracks, 1)))
        event_metrics[band_name]["event_mean_hits_lost_per_truth_track"].append(float(np.mean(num_hits_lost.astype(np.float64))) if num_truth_tracks > 0 else np.nan)
        event_metrics[band_name]["event_mean_hit_retention_per_truth_track"].append(float(np.mean(hit_retention.astype(np.float64))) if num_truth_tracks > 0 else np.nan)

        if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
            print(f"[{assignments.split}] track-loss events processed {event_idx + 1}/{len(assignments.band_codes)}")

    return BandTrackLossSamples(
        split=assignments.split,
        track_samples=track_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in event_metrics.items()
        },
        totals=totals,
    )


def _collect_band_feature_samples(
    cfg: dict,
    assignments: BandAssignments,
    hit_selection: str,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    rng_seed: int,
) -> BandFeatureSamples:
    if assignments.hit_eval_path is None:
        raise RuntimeError(f"Band feature analysis requires hit filtering, but split '{assignments.split}' has no hit_eval_path.")

    dataset_post, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=True)
    dataset_truth, _, _ = _build_dataset(cfg=cfg, split=assignments.split, use_hit_eval=False)
    rng = np.random.default_rng(rng_seed)

    hit_samples = {band_name: pd.DataFrame(columns=[*HIT_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    particle_samples = {band_name: pd.DataFrame(columns=[*PARTICLE_FEATURE_COLUMNS, "_sample_key"]) for band_name in BAND_NAMES}
    event_hits_per_track = {band_name: [] for band_name in BAND_NAMES}
    totals = {
        band_name: {
            "num_events": 0,
            "num_filtered_hits": 0,
            "num_selected_hits": 0,
            "num_reconstructable_tracks": 0,
        }
        for band_name in BAND_NAMES
    }

    print(
        f"[{assignments.split}] collecting band feature samples "
        f"(hit_selection={hit_selection}, max_hit_samples={max_hit_samples_per_band}, "
        f"max_particle_samples={max_particle_samples_per_band})"
    )
    for event_idx, band_code in enumerate(assignments.band_codes):
        band_name = BAND_NAMES[int(band_code)]
        hits_post, particles_post = dataset_post.load_event(event_idx)
        hits_truth, _ = dataset_truth.load_event(event_idx)

        if hit_selection == "all":
            selected_hits = hits_post
        elif hit_selection == "valid-track":
            selected_hits = hits_post[hits_post["on_valid_particle"]]
        else:
            raise ValueError(f"Unknown hit_selection={hit_selection!r}")

        particle_ids = particles_post["particle_id"]
        post_hit_counts = (
            hits_post[hits_post["on_valid_particle"]]
            .groupby("particle_id")
            .size()
            .reindex(particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        pre_hit_counts = (
            hits_truth[hits_truth["particle_id"].isin(particle_ids)]
            .groupby("particle_id")
            .size()
            .reindex(particle_ids, fill_value=0)
            .to_numpy(dtype=np.int32)
        )
        hit_retention = np.divide(
            post_hit_counts,
            pre_hit_counts,
            out=np.zeros_like(post_hit_counts, dtype=np.float64),
            where=pre_hit_counts > 0,
        )

        hit_frame = selected_hits.loc[:, list(HIT_FEATURE_COLUMNS)]
        particle_frame = particles_post.loc[:, ["pt", "eta", "phi"]].copy()
        particle_frame["num_hits_post"] = post_hit_counts
        particle_frame["num_hits_pre"] = pre_hit_counts
        particle_frame["hit_retention"] = hit_retention

        hit_samples[band_name] = _priority_sample(
            hit_samples[band_name],
            hit_frame,
            max_rows=max_hit_samples_per_band,
            rng=rng,
        )
        particle_samples[band_name] = _priority_sample(
            particle_samples[band_name],
            particle_frame,
            max_rows=max_particle_samples_per_band,
            rng=rng,
        )

        totals[band_name]["num_events"] += 1
        totals[band_name]["num_filtered_hits"] += int(len(hits_post))
        totals[band_name]["num_selected_hits"] += int(len(selected_hits))
        totals[band_name]["num_reconstructable_tracks"] += int(len(particles_post))
        event_hits_per_track[band_name].append(float(len(hits_post)) / max(float(len(particles_post)), 1.0))

        if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
            print(f"[{assignments.split}] band feature events processed {event_idx + 1}/{len(assignments.band_codes)}")

    return BandFeatureSamples(
        split=assignments.split,
        hit_selection=hit_selection,
        hit_samples=hit_samples,
        particle_samples=particle_samples,
        event_hits_per_track={k: np.asarray(v, dtype=np.float64) for k, v in event_hits_per_track.items()},
        totals=totals,
    )


def _feature_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    values = values.astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "min": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _combined_values(*arrays: np.ndarray) -> np.ndarray:
    kept = [array.astype(np.float64)[np.isfinite(array.astype(np.float64))] for array in arrays if array.size > 0]
    if not kept:
        return np.array([], dtype=np.float64)
    return np.concatenate(kept)


def _continuous_bins(values: np.ndarray, n_bins: int = 60, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.min(values))
        hi = float(np.max(values))
    if np.isclose(lo, hi):
        delta = max(1e-6, abs(lo) * 0.05 + 1e-6)
        lo -= delta
        hi += delta
    return np.linspace(lo, hi, n_bins + 1)


def _count_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.arange(-0.5, 1.5, 1.0)

    lo = int(np.floor(np.min(values)))
    hi = int(np.ceil(np.percentile(values, 99.5)))
    hi = max(hi, lo + 1)
    if (hi - lo) <= 60:
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    return np.linspace(lo, hi, 61)


def _feature_bins_and_scale(feature_key: str, values: np.ndarray) -> tuple[np.ndarray, str]:
    if feature_key in {"hit_phi", "particle_phi"}:
        return np.linspace(-np.pi, np.pi, 61), "linear"
    if feature_key == "particle_pt":
        positive = values[values > 0]
        if positive.size == 0:
            return np.linspace(0.0, 1.0, 61), "linear"
        lo = float(np.percentile(positive, 0.5))
        hi = float(np.percentile(positive, 99.5))
        lo = max(lo, float(np.min(positive)))
        hi = max(hi, lo * 1.05)
        return np.geomspace(lo, hi, 61), "log"
    if feature_key in {"particle_num_hits_post", "particle_num_hits_pre"}:
        return _count_bins(values), "linear"
    if feature_key == "particle_hit_retention":
        return np.linspace(0.0, 1.0, 51), "linear"
    return _continuous_bins(values), "linear"


def _feature_map(samples: BandFeatureSamples, band_name: str) -> dict[str, np.ndarray]:
    hit_df = samples.hit_samples[band_name]
    particle_df = samples.particle_samples[band_name]
    return {
        "event_hits_per_track": samples.event_hits_per_track[band_name],
        "hit_x": hit_df["x"].to_numpy(dtype=np.float64) if "x" in hit_df else np.array([], dtype=np.float64),
        "hit_y": hit_df["y"].to_numpy(dtype=np.float64) if "y" in hit_df else np.array([], dtype=np.float64),
        "hit_z": hit_df["z"].to_numpy(dtype=np.float64) if "z" in hit_df else np.array([], dtype=np.float64),
        "hit_r": hit_df["r"].to_numpy(dtype=np.float64) if "r" in hit_df else np.array([], dtype=np.float64),
        "hit_eta": hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in hit_df else np.array([], dtype=np.float64),
        "hit_phi": hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in hit_df else np.array([], dtype=np.float64),
        "particle_pt": particle_df["pt"].to_numpy(dtype=np.float64) if "pt" in particle_df else np.array([], dtype=np.float64),
        "particle_eta": particle_df["eta"].to_numpy(dtype=np.float64) if "eta" in particle_df else np.array([], dtype=np.float64),
        "particle_phi": particle_df["phi"].to_numpy(dtype=np.float64) if "phi" in particle_df else np.array([], dtype=np.float64),
        "particle_num_hits_post": (
            particle_df["num_hits_post"].to_numpy(dtype=np.float64) if "num_hits_post" in particle_df else np.array([], dtype=np.float64)
        ),
        "particle_num_hits_pre": (
            particle_df["num_hits_pre"].to_numpy(dtype=np.float64) if "num_hits_pre" in particle_df else np.array([], dtype=np.float64)
        ),
        "particle_hit_retention": (
            particle_df["hit_retention"].to_numpy(dtype=np.float64) if "hit_retention" in particle_df else np.array([], dtype=np.float64)
        ),
    }


def _truth_feature_map(samples: TruthBandSamples, band_name: str) -> dict[str, np.ndarray]:
    hit_df = samples.hit_samples[band_name]
    track_df = samples.track_samples[band_name]
    metrics = samples.event_metrics[band_name]
    return {
        "event_true_tracks": metrics["event_true_tracks"],
        "event_true_hits": metrics["event_true_hits"],
        "event_true_hits_per_true_track": metrics["event_true_hits_per_true_track"],
        "true_hit_x": hit_df["x"].to_numpy(dtype=np.float64) if "x" in hit_df else np.array([], dtype=np.float64),
        "true_hit_y": hit_df["y"].to_numpy(dtype=np.float64) if "y" in hit_df else np.array([], dtype=np.float64),
        "true_hit_z": hit_df["z"].to_numpy(dtype=np.float64) if "z" in hit_df else np.array([], dtype=np.float64),
        "true_hit_r": hit_df["r"].to_numpy(dtype=np.float64) if "r" in hit_df else np.array([], dtype=np.float64),
        "true_hit_eta": hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in hit_df else np.array([], dtype=np.float64),
        "true_hit_phi": hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in hit_df else np.array([], dtype=np.float64),
        "true_track_pt": track_df["pt"].to_numpy(dtype=np.float64) if "pt" in track_df else np.array([], dtype=np.float64),
        "true_track_eta": track_df["eta"].to_numpy(dtype=np.float64) if "eta" in track_df else np.array([], dtype=np.float64),
        "true_track_phi": track_df["phi"].to_numpy(dtype=np.float64) if "phi" in track_df else np.array([], dtype=np.float64),
        "true_track_num_hits": (
            track_df["num_true_hits"].to_numpy(dtype=np.float64) if "num_true_hits" in track_df else np.array([], dtype=np.float64)
        ),
    }


def _track_loss_feature_map(samples: BandTrackLossSamples, band_name: str) -> dict[str, np.ndarray]:
    track_df = samples.track_samples[band_name]
    metrics = samples.event_metrics[band_name]
    return {
        "event_truth_tracks": metrics["event_truth_tracks"],
        "event_surviving_tracks": metrics["event_surviving_tracks"],
        "event_lost_tracks": metrics["event_lost_tracks"],
        "event_lost_track_fraction": metrics["event_lost_track_fraction"],
        "event_mean_hits_lost_per_truth_track": metrics["event_mean_hits_lost_per_truth_track"],
        "event_mean_hit_retention_per_truth_track": metrics["event_mean_hit_retention_per_truth_track"],
        "track_pt": track_df["pt"].to_numpy(dtype=np.float64) if "pt" in track_df else np.array([], dtype=np.float64),
        "track_eta": track_df["eta"].to_numpy(dtype=np.float64) if "eta" in track_df else np.array([], dtype=np.float64),
        "track_phi": track_df["phi"].to_numpy(dtype=np.float64) if "phi" in track_df else np.array([], dtype=np.float64),
        "track_num_hits_pre": track_df["num_hits_pre"].to_numpy(dtype=np.float64) if "num_hits_pre" in track_df else np.array([], dtype=np.float64),
        "track_num_hits_post": track_df["num_hits_post"].to_numpy(dtype=np.float64) if "num_hits_post" in track_df else np.array([], dtype=np.float64),
        "track_num_hits_lost": track_df["num_hits_lost"].to_numpy(dtype=np.float64) if "num_hits_lost" in track_df else np.array([], dtype=np.float64),
        "track_hit_retention": track_df["hit_retention"].to_numpy(dtype=np.float64) if "hit_retention" in track_df else np.array([], dtype=np.float64),
        "track_survived_filter": track_df["survived_filter"].to_numpy(dtype=np.float64) if "survived_filter" in track_df else np.array([], dtype=np.float64),
    }


def _compose_fail_reason_labels(
    fail_pt: np.ndarray,
    fail_eta: np.ndarray,
    fail_min_hits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fail_reason = np.full(len(fail_pt), "valid", dtype=object)
    primary_fail_reason = np.full(len(fail_pt), "valid", dtype=object)
    for idx in range(len(fail_pt)):
        reasons: list[str] = []
        if bool(fail_pt[idx]):
            reasons.append("pt")
        if bool(fail_eta[idx]):
            reasons.append("eta")
        if bool(fail_min_hits[idx]):
            reasons.append("min_hits")
        if reasons:
            fail_reason[idx] = "+".join(reasons)
            primary_fail_reason[idx] = reasons[0] if len(reasons) == 1 else "multiple"
    return fail_reason.astype(str), primary_fail_reason.astype(str)


def _collect_band_raw_event_sample_bundle(
    cfg: dict,
    assignments: BandAssignments,
    *,
    collect_invalid_source: bool,
    collect_prefilter_lowpt_noise: bool,
    collect_prefilter_truth_nonreco: bool,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    rng_seed: int,
    crowding_distance: float,
) -> tuple[BandInvalidSourceSamples | None, BandPrefilterLowPtNoiseSamples | None, BandPrefilterTruthNonrecoSamples | None]:
    import h5py

    if not any((collect_invalid_source, collect_prefilter_lowpt_noise, collect_prefilter_truth_nonreco)):
        return None, None, None
    if assignments.hit_eval_path is None and (collect_invalid_source or collect_prefilter_lowpt_noise):
        raise RuntimeError(
            f"Raw-event bundle requires hit filtering for split '{assignments.split}', but no hit_eval_path is available."
        )

    data_cfg = cfg["data"]
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    min_pt = float(data_cfg["particle_min_pt"])
    max_abs_eta = float(data_cfg["particle_max_abs_eta"])
    min_num_hits = int(data_cfg["particle_min_num_hits"])
    hit_threshold = float(data_cfg.get("hit_filter_threshold", 0.1))
    rng_invalid = np.random.default_rng(rng_seed + 3000)
    rng_lowpt = np.random.default_rng(rng_seed + 4000)
    rng_truth = np.random.default_rng(rng_seed + 5000)

    invalid_source_samples: BandInvalidSourceSamples | None = None
    prefilter_lowpt_noise_samples: BandPrefilterLowPtNoiseSamples | None = None
    prefilter_truth_nonreco_samples: BandPrefilterTruthNonrecoSamples | None = None

    if collect_invalid_source:
        invalid_nonreco_hit_samples = {
            band_name: pd.DataFrame(
                columns=[
                    "x",
                    "y",
                    "z",
                    "r",
                    "eta",
                    "phi",
                    "score",
                    "min_distance_to_true_same_layer",
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_hits_post",
                    "parent_num_kept_invalid_hits",
                    "parent_fail_reason",
                    "parent_primary_fail_reason",
                    "_sample_key",
                ]
            )
            for band_name in BAND_NAMES
        }
        invalid_nonreco_particle_samples = {
            band_name: pd.DataFrame(
                columns=[
                    "pt",
                    "eta",
                    "phi",
                    "num_hits_pre",
                    "num_hits_post",
                    "num_kept_invalid_hits",
                    "fail_pt",
                    "fail_eta",
                    "fail_min_hits",
                    "fail_reason",
                    "primary_fail_reason",
                    "_sample_key",
                ]
            )
            for band_name in BAND_NAMES
        }
        invalid_noise_hit_samples = {
            band_name: pd.DataFrame(
                columns=["x", "y", "z", "r", "eta", "phi", "score", "min_distance_to_true_same_layer", "volume_id", "layer_id", "_sample_key"]
            )
            for band_name in BAND_NAMES
        }
        invalid_event_metrics = {
            band_name: {
                "event_kept_nonreco_hits": [],
                "event_kept_noise_hits": [],
                "event_nonreco_source_particles": [],
                "event_fraction_invalid_nonreco": [],
                "event_fraction_invalid_noise": [],
                "event_kept_nonreco_hits_fail_pt": [],
                "event_kept_nonreco_hits_fail_eta": [],
                "event_kept_nonreco_hits_fail_min_hits": [],
            }
            for band_name in BAND_NAMES
        }
        invalid_totals = {
            band_name: {
                "num_events": 0,
                "num_kept_nonreco_hits": 0,
                "num_kept_noise_hits": 0,
                "num_nonreco_source_particles": 0,
            }
            for band_name in BAND_NAMES
        }
        invalid_failure_rows: list[dict[str, float | int | str]] = []

    if collect_prefilter_lowpt_noise:
        lowpt_hit_samples = {
            band_name: pd.DataFrame(
                columns=[
                    "x",
                    "y",
                    "z",
                    "r",
                    "eta",
                    "phi",
                    "score",
                    "filter_status",
                    "min_distance_to_valid_same_layer",
                    "proximity_bucket",
                    "volume_id",
                    "layer_id",
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_lowpt_hits_total",
                    "parent_num_lowpt_hits_kept",
                    "parent_num_lowpt_hits_removed",
                    "parent_kept_hit_fraction",
                    "_sample_key",
                ]
            )
            for band_name in BAND_NAMES
        }
        lowpt_particle_samples = {
            band_name: pd.DataFrame(
                columns=[
                    "pt",
                    "eta",
                    "phi",
                    "num_hits_pre",
                    "num_lowpt_hits_total",
                    "num_lowpt_hits_kept",
                    "num_lowpt_hits_removed",
                    "kept_hit_fraction",
                    "_sample_key",
                ]
            )
            for band_name in BAND_NAMES
        }
        lowpt_noise_hit_samples = {
            band_name: pd.DataFrame(
                columns=["x", "y", "z", "r", "eta", "phi", "score", "filter_status", "volume_id", "layer_id", "_sample_key"]
            )
            for band_name in BAND_NAMES
        }
        lowpt_event_metrics = {
            band_name: {
                "event_prefilter_lowpt_hits": [],
                "event_prefilter_noise_hits": [],
                "event_fraction_prefilter_lowpt": [],
                "event_fraction_prefilter_noise": [],
                "event_total_raw_hits": [],
                "event_total_valid_raw_hits": [],
                "event_prefilter_lowpt_source_particles": [],
                "event_fraction_all_hits_lowpt": [],
                "event_fraction_all_hits_noise": [],
                "event_prefilter_lowpt_hits_kept": [],
                "event_prefilter_lowpt_hits_removed": [],
                "event_prefilter_noise_hits_kept": [],
                "event_prefilter_noise_hits_removed": [],
                "event_prefilter_lowpt_near_valid_hits": [],
                "event_prefilter_lowpt_isolated_hits": [],
                "event_lowpt_hit_near_valid_fraction": [],
                "event_lowpt_hit_kept_fraction": [],
                "event_lowpt_hit_kept_fraction_near_valid": [],
                "event_lowpt_hit_kept_fraction_isolated": [],
                "event_valid_hit_recall": [],
                "event_filtered_contamination": [],
            }
            for band_name in BAND_NAMES
        }
        lowpt_totals = {
            band_name: {
                "num_events": 0,
                "num_prefilter_lowpt_hits": 0,
                "num_prefilter_noise_hits": 0,
                "num_lowpt_source_particles": 0,
            }
            for band_name in BAND_NAMES
        }
        lowpt_event_rows: list[dict[str, float | int | str]] = []
        lowpt_region_accumulators: dict[tuple[str, int, int], dict[str, float]] = {}

    if collect_prefilter_truth_nonreco:
        truth_nonreco_particle_samples = {
            band_name: pd.DataFrame(
                columns=["pt", "eta", "phi", "num_hits_pre", "num_unique_layers", "primary_fail_reason", "fail_reason", "_sample_key"]
            )
            for band_name in BAND_NAMES
        }
        truth_nonreco_event_metrics = {
            band_name: {
                "event_total_nonreco": [],
                "event_total_particles": [],
                "event_num_fail_pt": [],
                "event_num_fail_contains_pt": [],
                "event_num_fail_eta": [],
                "event_num_fail_min_hits": [],
                "event_num_fail_multiple": [],
                "event_fraction_nonreco": [],
            }
            for band_name in BAND_NAMES
        }
        truth_nonreco_totals = {
            band_name: {
                "num_events": 0,
                "num_nonreco_particles": 0,
                "num_fail_pt": 0,
                "num_fail_contains_pt": 0,
                "num_fail_eta": 0,
                "num_fail_min_hits": 0,
                "num_fail_multiple": 0,
            }
            for band_name in BAND_NAMES
        }
        truth_nonreco_noise_hit_samples = {
            band_name: pd.DataFrame(columns=["r", "eta", "phi", "z", "_sample_key"])
            for band_name in BAND_NAMES
        }
        truth_nonreco_noise_event_metrics = {
            band_name: {"event_total_noise_hits": [], "event_total_hits": [], "event_noise_fraction": []}
            for band_name in BAND_NAMES
        }

    enabled_sections = []
    if collect_invalid_source:
        enabled_sections.append("invalid-source")
    if collect_prefilter_lowpt_noise:
        enabled_sections.append("prefilter-lowpt-noise")
    if collect_prefilter_truth_nonreco:
        enabled_sections.append("prefilter-truth-nonreco")
    print(f"[{assignments.split}] collecting raw-event bundle: {', '.join(enabled_sections)}")

    hit_eval_file = h5py.File(assignments.hit_eval_path, "r") if (collect_invalid_source or collect_prefilter_lowpt_noise) else None
    try:
        for event_idx, band_code in enumerate(assignments.band_codes):
            band_name = BAND_NAMES[int(band_code)]
            hits_raw, particles_raw = _load_raw_trackml_event(assignments.split_dir, assignments.event_names[event_idx], hit_volume_ids)

            hit_counts_pre = hits_raw["particle_id"].value_counts()
            hit_particle_ids = hits_raw["particle_id"].to_numpy(dtype=np.int64, copy=False)
            particles_pre = particles_raw.copy()
            particles_pre["num_hits_pre"] = particles_pre["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)

            keep_hits = None
            hits_scored = None
            hits_kept = None
            particles_post = None
            valid_hits_kept = None
            if hit_eval_file is not None:
                hit_scores = _read_hit_filter_scores(hit_eval_file, int(assignments.sample_ids[event_idx]))
                if hit_scores is None or len(hit_scores) != len(hits_raw):
                    raise RuntimeError(
                        f"Missing or misaligned hit-filter scores for split '{assignments.split}', sample_id={assignments.sample_ids[event_idx]}."
                    )
                score_array = np.asarray(hit_scores, dtype=np.float64)
                keep_hits = score_array >= hit_threshold
                hits_scored = hits_raw.copy()
                hits_scored["score"] = score_array
                hits_scored["filter_status"] = np.where(keep_hits, "kept", "removed")
                hits_kept = hits_scored.loc[keep_hits, :].copy()
                hit_counts_post = hits_kept["particle_id"].value_counts()
                particles_post = particles_pre.copy()
                particles_post["num_hits_post"] = particles_post["particle_id"].map(hit_counts_post).fillna(0).astype(np.int32)

            if collect_prefilter_truth_nonreco:
                unique_layer_counts = (
                    hits_raw.assign(_lk=hits_raw["volume_id"] * 1000 + hits_raw["layer_id"])
                    .groupby("particle_id")["_lk"]
                    .nunique()
                )
                particles_truth_nonreco = particles_pre.copy()
                particles_truth_nonreco["num_unique_layers"] = particles_truth_nonreco["particle_id"].map(unique_layer_counts).fillna(0).astype(np.int32)

                fail_pt_pre = particles_truth_nonreco["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt
                fail_eta_pre = np.abs(particles_truth_nonreco["eta"].to_numpy(dtype=np.float64, copy=False)) >= max_abs_eta
                fail_min_hits_pre = particles_truth_nonreco["num_hits_pre"].to_numpy(dtype=np.int32, copy=False) < min_num_hits
                fail_reason_pre, primary_fail_reason_pre = _compose_fail_reason_labels(fail_pt_pre, fail_eta_pre, fail_min_hits_pre)
                particles_truth_nonreco["fail_reason"] = fail_reason_pre
                particles_truth_nonreco["primary_fail_reason"] = primary_fail_reason_pre

                is_nonreco = fail_pt_pre | fail_eta_pre | fail_min_hits_pre
                nonreco_particles = particles_truth_nonreco.loc[
                    is_nonreco, ["pt", "eta", "phi", "num_hits_pre", "num_unique_layers", "primary_fail_reason", "fail_reason"]
                ].copy()
                prim = nonreco_particles["primary_fail_reason"].to_numpy(dtype=str, copy=False)
                num_total = len(particles_truth_nonreco)
                num_nonreco = int(is_nonreco.sum())
                num_fail_pt = int(np.sum(prim == "pt"))
                num_fail_contains_pt = int(np.sum(fail_pt_pre))
                num_fail_eta = int(np.sum(prim == "eta"))
                num_fail_min_hits = int(np.sum(prim == "min_hits"))
                num_fail_multiple = int(np.sum(prim == "multiple"))

                truth_nonreco_event_metrics[band_name]["event_total_nonreco"].append(num_nonreco)
                truth_nonreco_event_metrics[band_name]["event_total_particles"].append(num_total)
                truth_nonreco_event_metrics[band_name]["event_num_fail_pt"].append(num_fail_pt)
                truth_nonreco_event_metrics[band_name]["event_num_fail_contains_pt"].append(num_fail_contains_pt)
                truth_nonreco_event_metrics[band_name]["event_num_fail_eta"].append(num_fail_eta)
                truth_nonreco_event_metrics[band_name]["event_num_fail_min_hits"].append(num_fail_min_hits)
                truth_nonreco_event_metrics[band_name]["event_num_fail_multiple"].append(num_fail_multiple)
                truth_nonreco_event_metrics[band_name]["event_fraction_nonreco"].append(float(num_nonreco) / float(num_total) if num_total > 0 else 0.0)
                truth_nonreco_totals[band_name]["num_events"] += 1
                truth_nonreco_totals[band_name]["num_nonreco_particles"] += num_nonreco
                truth_nonreco_totals[band_name]["num_fail_pt"] += num_fail_pt
                truth_nonreco_totals[band_name]["num_fail_contains_pt"] += num_fail_contains_pt
                truth_nonreco_totals[band_name]["num_fail_eta"] += num_fail_eta
                truth_nonreco_totals[band_name]["num_fail_min_hits"] += num_fail_min_hits
                truth_nonreco_totals[band_name]["num_fail_multiple"] += num_fail_multiple
                truth_nonreco_particle_samples[band_name] = _priority_sample(
                    truth_nonreco_particle_samples[band_name],
                    nonreco_particles,
                    max_particle_samples_per_band,
                    rng_truth,
                )

                truth_noise_hits = hits_raw.loc[hits_raw["particle_id"] == 0, ["r", "eta", "phi", "z"]].copy()
                truth_nonreco_noise_event_metrics[band_name]["event_total_noise_hits"].append(len(truth_noise_hits))
                truth_nonreco_noise_event_metrics[band_name]["event_total_hits"].append(len(hits_raw))
                truth_nonreco_noise_event_metrics[band_name]["event_noise_fraction"].append(float(len(truth_noise_hits)) / float(len(hits_raw)) if len(hits_raw) > 0 else 0.0)
                truth_nonreco_noise_hit_samples[band_name] = _priority_sample(
                    truth_nonreco_noise_hit_samples[band_name],
                    truth_noise_hits,
                    max_hit_samples_per_band,
                    rng_truth,
                )

            if collect_invalid_source:
                assert hits_kept is not None and particles_post is not None and hits_scored is not None
                fail_pt_post = particles_post["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt
                fail_eta_post = np.abs(particles_post["eta"].to_numpy(dtype=np.float64, copy=False)) >= max_abs_eta
                fail_min_hits_post = particles_post["num_hits_post"].to_numpy(dtype=np.int32, copy=False) < min_num_hits
                fail_reason_post, primary_fail_reason_post = _compose_fail_reason_labels(fail_pt_post, fail_eta_post, fail_min_hits_post)
                particles_invalid = particles_post.copy()
                particles_invalid["fail_pt"] = fail_pt_post
                particles_invalid["fail_eta"] = fail_eta_post
                particles_invalid["fail_min_hits"] = fail_min_hits_post
                particles_invalid["fail_reason"] = fail_reason_post
                particles_invalid["primary_fail_reason"] = primary_fail_reason_post

                valid_particle_ids_post = particles_invalid.loc[~(fail_pt_post | fail_eta_post | fail_min_hits_post), "particle_id"].to_numpy(dtype=np.int64, copy=False)
                hits_kept["on_rebuilt_valid_particle"] = hits_kept["particle_id"].isin(valid_particle_ids_post)
                kept_invalid_hits = hits_kept.loc[~hits_kept["on_rebuilt_valid_particle"], :].copy()
                kept_nonreco_hits = kept_invalid_hits.loc[kept_invalid_hits["particle_id"] != 0, :].copy()
                kept_noise_hits = kept_invalid_hits.loc[kept_invalid_hits["particle_id"] == 0, :].copy()
                valid_hits_kept = hits_kept.loc[hits_kept["on_rebuilt_valid_particle"], :]

                nonreco_min_dist = _same_layer_min_rphiz_distances(kept_nonreco_hits, valid_hits_kept)
                noise_min_dist = _same_layer_min_rphiz_distances(kept_noise_hits, valid_hits_kept)
                kept_nonreco_hits["min_distance_to_true_same_layer"] = nonreco_min_dist
                kept_noise_hits["min_distance_to_true_same_layer"] = noise_min_dist

                nonreco_particles_post = particles_invalid.loc[particles_invalid["particle_id"].isin(kept_nonreco_hits["particle_id"].unique()), :].copy()
                kept_nonreco_counts = kept_nonreco_hits["particle_id"].value_counts()
                nonreco_particles_post["num_kept_invalid_hits"] = nonreco_particles_post["particle_id"].map(kept_nonreco_counts).fillna(0).astype(np.int32)

                parent_columns = [
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_hits_post",
                    "parent_num_kept_invalid_hits",
                    "parent_fail_reason",
                    "parent_primary_fail_reason",
                ]
                if not kept_nonreco_hits.empty:
                    parent_lookup = nonreco_particles_post.set_index("particle_id")[
                        [
                            "pt",
                            "eta",
                            "phi",
                            "num_hits_pre",
                            "num_hits_post",
                            "num_kept_invalid_hits",
                            "fail_reason",
                            "primary_fail_reason",
                        ]
                    ].rename(
                        columns={
                            "pt": "parent_pt",
                            "eta": "parent_eta",
                            "phi": "parent_phi",
                            "num_hits_pre": "parent_num_hits_pre",
                            "num_hits_post": "parent_num_hits_post",
                            "num_kept_invalid_hits": "parent_num_kept_invalid_hits",
                            "fail_reason": "parent_fail_reason",
                            "primary_fail_reason": "parent_primary_fail_reason",
                        }
                    )
                    kept_nonreco_hits = kept_nonreco_hits.join(parent_lookup, on="particle_id")
                for column in parent_columns:
                    if column not in kept_nonreco_hits.columns:
                        kept_nonreco_hits[column] = np.nan

                invalid_nonreco_hit_frame = kept_nonreco_hits.loc[
                    :,
                    [
                        "x",
                        "y",
                        "z",
                        "r",
                        "eta",
                        "phi",
                        "score",
                        "min_distance_to_true_same_layer",
                        "parent_pt",
                        "parent_eta",
                        "parent_phi",
                        "parent_num_hits_pre",
                        "parent_num_hits_post",
                        "parent_num_kept_invalid_hits",
                        "parent_fail_reason",
                        "parent_primary_fail_reason",
                    ],
                ]
                invalid_nonreco_particle_frame = nonreco_particles_post.loc[
                    :,
                    [
                        "pt",
                        "eta",
                        "phi",
                        "num_hits_pre",
                        "num_hits_post",
                        "num_kept_invalid_hits",
                        "fail_pt",
                        "fail_eta",
                        "fail_min_hits",
                        "fail_reason",
                        "primary_fail_reason",
                    ],
                ]
                invalid_noise_hit_frame = kept_noise_hits.loc[
                    :,
                    ["x", "y", "z", "r", "eta", "phi", "score", "min_distance_to_true_same_layer", "volume_id", "layer_id"],
                ]

                invalid_nonreco_hit_samples[band_name] = _priority_sample(
                    invalid_nonreco_hit_samples[band_name], invalid_nonreco_hit_frame, max_hit_samples_per_band, rng_invalid
                )
                invalid_nonreco_particle_samples[band_name] = _priority_sample(
                    invalid_nonreco_particle_samples[band_name], invalid_nonreco_particle_frame, max_particle_samples_per_band, rng_invalid
                )
                invalid_noise_hit_samples[band_name] = _priority_sample(
                    invalid_noise_hit_samples[band_name], invalid_noise_hit_frame, max_hit_samples_per_band, rng_invalid
                )

                num_invalid = max(len(kept_invalid_hits), 1)
                invalid_event_metrics[band_name]["event_kept_nonreco_hits"].append(int(len(kept_nonreco_hits)))
                invalid_event_metrics[band_name]["event_kept_noise_hits"].append(int(len(kept_noise_hits)))
                invalid_event_metrics[band_name]["event_nonreco_source_particles"].append(int(len(nonreco_particles_post)))
                invalid_event_metrics[band_name]["event_fraction_invalid_nonreco"].append(float(len(kept_nonreco_hits) / num_invalid))
                invalid_event_metrics[band_name]["event_fraction_invalid_noise"].append(float(len(kept_noise_hits) / num_invalid))
                invalid_event_metrics[band_name]["event_kept_nonreco_hits_fail_pt"].append(
                    int(np.sum(kept_nonreco_hits["parent_pt"].to_numpy(dtype=np.float64) <= min_pt)) if not kept_nonreco_hits.empty else 0
                )
                invalid_event_metrics[band_name]["event_kept_nonreco_hits_fail_eta"].append(
                    int(np.sum(np.abs(kept_nonreco_hits["parent_eta"].to_numpy(dtype=np.float64)) >= max_abs_eta)) if not kept_nonreco_hits.empty else 0
                )
                invalid_event_metrics[band_name]["event_kept_nonreco_hits_fail_min_hits"].append(
                    int(np.sum(kept_nonreco_hits["parent_num_hits_post"].to_numpy(dtype=np.int32) < min_num_hits)) if not kept_nonreco_hits.empty else 0
                )

                invalid_totals[band_name]["num_events"] += 1
                invalid_totals[band_name]["num_kept_nonreco_hits"] += int(len(kept_nonreco_hits))
                invalid_totals[band_name]["num_kept_noise_hits"] += int(len(kept_noise_hits))
                invalid_totals[band_name]["num_nonreco_source_particles"] += int(len(nonreco_particles_post))

                for fail_label in sorted(nonreco_particles_post["primary_fail_reason"].astype(str).value_counts().index):
                    part_count = int(np.sum(nonreco_particles_post["primary_fail_reason"].astype(str).to_numpy() == fail_label))
                    hit_count = int(np.sum(kept_nonreco_hits["parent_primary_fail_reason"].astype(str).to_numpy() == fail_label)) if not kept_nonreco_hits.empty else 0
                    invalid_failure_rows.append(
                        {
                            "split": assignments.split,
                            "band": band_name,
                            "primary_fail_reason": fail_label,
                            "num_particles": part_count,
                            "num_hits": hit_count,
                        }
                    )

            if collect_prefilter_lowpt_noise:
                assert hits_scored is not None and keep_hits is not None and particles_post is not None
                valid_particle_mask_pre = (
                    (particles_pre["pt"].to_numpy(dtype=np.float64, copy=False) > min_pt)
                    & (np.abs(particles_pre["eta"].to_numpy(dtype=np.float64, copy=False)) < max_abs_eta)
                    & (particles_pre["num_hits_pre"].to_numpy(dtype=np.int32, copy=False) >= min_num_hits)
                )
                valid_particle_ids_pre = particles_pre.loc[valid_particle_mask_pre, "particle_id"].to_numpy(dtype=np.int64, copy=False)
                valid_hits_raw = hits_scored.loc[hits_scored["particle_id"].isin(valid_particle_ids_pre), :].copy()

                lowpt_particle_ids = particles_pre.loc[
                    particles_pre["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt,
                    "particle_id",
                ].to_numpy(dtype=np.int64, copy=False)
                lowpt_hits = hits_scored.loc[hits_scored["particle_id"].isin(lowpt_particle_ids), :].copy()
                lowpt_noise_hits_raw = hits_scored.loc[hits_scored["particle_id"] == 0, :].copy()
                lowpt_min_distances = _same_layer_min_rphiz_distances(lowpt_hits, valid_hits_raw)
                lowpt_hits["min_distance_to_valid_same_layer"] = lowpt_min_distances
                lowpt_hits["proximity_bucket"] = np.where(
                    np.isfinite(lowpt_min_distances) & (lowpt_min_distances < crowding_distance),
                    "near_valid",
                    "isolated",
                )

                lowpt_particles = particles_pre.loc[particles_pre["particle_id"].isin(lowpt_hits["particle_id"].unique()), :].copy()
                lowpt_hit_counts_total = lowpt_hits["particle_id"].value_counts()
                lowpt_hit_counts_kept = lowpt_hits.loc[lowpt_hits["filter_status"] == "kept", "particle_id"].value_counts()
                lowpt_particles["num_lowpt_hits_total"] = lowpt_particles["particle_id"].map(lowpt_hit_counts_total).fillna(0).astype(np.int32)
                lowpt_particles["num_lowpt_hits_kept"] = lowpt_particles["particle_id"].map(lowpt_hit_counts_kept).fillna(0).astype(np.int32)
                lowpt_particles["num_lowpt_hits_removed"] = (
                    lowpt_particles["num_lowpt_hits_total"].to_numpy(dtype=np.int32, copy=False)
                    - lowpt_particles["num_lowpt_hits_kept"].to_numpy(dtype=np.int32, copy=False)
                )
                lowpt_particles["kept_hit_fraction"] = np.divide(
                    lowpt_particles["num_lowpt_hits_kept"].to_numpy(dtype=np.float64, copy=False),
                    np.maximum(lowpt_particles["num_lowpt_hits_total"].to_numpy(dtype=np.float64, copy=False), 1.0),
                )

                parent_columns = [
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_lowpt_hits_total",
                    "parent_num_lowpt_hits_kept",
                    "parent_num_lowpt_hits_removed",
                    "parent_kept_hit_fraction",
                ]
                if not lowpt_hits.empty:
                    parent_lookup = lowpt_particles.set_index("particle_id")[
                        [
                            "pt",
                            "eta",
                            "phi",
                            "num_hits_pre",
                            "num_lowpt_hits_total",
                            "num_lowpt_hits_kept",
                            "num_lowpt_hits_removed",
                            "kept_hit_fraction",
                        ]
                    ].rename(
                        columns={
                            "pt": "parent_pt",
                            "eta": "parent_eta",
                            "phi": "parent_phi",
                            "num_hits_pre": "parent_num_hits_pre",
                            "num_lowpt_hits_total": "parent_num_lowpt_hits_total",
                            "num_lowpt_hits_kept": "parent_num_lowpt_hits_kept",
                            "num_lowpt_hits_removed": "parent_num_lowpt_hits_removed",
                            "kept_hit_fraction": "parent_kept_hit_fraction",
                        }
                    )
                    lowpt_hits = lowpt_hits.join(parent_lookup, on="particle_id")
                for column in parent_columns:
                    if column not in lowpt_hits.columns:
                        lowpt_hits[column] = np.nan

                lowpt_hit_frame = lowpt_hits.loc[
                    :,
                    [
                        "x",
                        "y",
                        "z",
                        "r",
                        "eta",
                        "phi",
                        "score",
                        "filter_status",
                        "min_distance_to_valid_same_layer",
                        "proximity_bucket",
                        "volume_id",
                        "layer_id",
                        "parent_pt",
                        "parent_eta",
                        "parent_phi",
                        "parent_num_hits_pre",
                        "parent_num_lowpt_hits_total",
                        "parent_num_lowpt_hits_kept",
                        "parent_num_lowpt_hits_removed",
                        "parent_kept_hit_fraction",
                    ],
                ]
                lowpt_particle_frame = lowpt_particles.loc[
                    :,
                    [
                        "pt",
                        "eta",
                        "phi",
                        "num_hits_pre",
                        "num_lowpt_hits_total",
                        "num_lowpt_hits_kept",
                        "num_lowpt_hits_removed",
                        "kept_hit_fraction",
                    ],
                ]
                lowpt_noise_hit_frame = lowpt_noise_hits_raw.loc[
                    :,
                    ["x", "y", "z", "r", "eta", "phi", "score", "filter_status", "volume_id", "layer_id"],
                ]

                lowpt_hit_samples[band_name] = _priority_sample(lowpt_hit_samples[band_name], lowpt_hit_frame, max_hit_samples_per_band, rng_lowpt)
                lowpt_particle_samples[band_name] = _priority_sample(lowpt_particle_samples[band_name], lowpt_particle_frame, max_particle_samples_per_band, rng_lowpt)
                lowpt_noise_hit_samples[band_name] = _priority_sample(lowpt_noise_hit_samples[band_name], lowpt_noise_hit_frame, max_hit_samples_per_band, rng_lowpt)

                total_prefilter_source_hits = max(int(len(lowpt_hits) + len(lowpt_noise_hits_raw)), 1)
                total_raw_hits = int(len(hits_scored))
                total_valid_raw_hits = int(len(valid_hits_raw))
                total_kept_hits = int(np.sum(keep_hits))
                raw_valid_particle_mask = np.isin(hit_particle_ids, valid_particle_ids_pre)
                total_kept_invalid_raw_hits = int(np.sum((~raw_valid_particle_mask) & keep_hits))
                lowpt_near_valid_mask = lowpt_hits["proximity_bucket"].to_numpy(dtype=str) == "near_valid" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
                lowpt_kept_mask = lowpt_hits["filter_status"].to_numpy(dtype=str) == "kept" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
                lowpt_isolated_mask = lowpt_hits["proximity_bucket"].to_numpy(dtype=str) == "isolated" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
                lowpt_near_valid_hits = int(np.sum(lowpt_near_valid_mask))
                lowpt_isolated_hits = int(np.sum(lowpt_isolated_mask))
                lowpt_kept_hits = int(np.sum(lowpt_kept_mask))
                valid_kept_hits = int(np.sum(raw_valid_particle_mask & keep_hits))

                lowpt_event_metrics[band_name]["event_prefilter_lowpt_hits"].append(int(len(lowpt_hits)))
                lowpt_event_metrics[band_name]["event_prefilter_noise_hits"].append(int(len(lowpt_noise_hits_raw)))
                lowpt_event_metrics[band_name]["event_fraction_prefilter_lowpt"].append(float(len(lowpt_hits) / total_prefilter_source_hits))
                lowpt_event_metrics[band_name]["event_fraction_prefilter_noise"].append(float(len(lowpt_noise_hits_raw) / total_prefilter_source_hits))
                lowpt_event_metrics[band_name]["event_total_raw_hits"].append(total_raw_hits)
                lowpt_event_metrics[band_name]["event_total_valid_raw_hits"].append(total_valid_raw_hits)
                lowpt_event_metrics[band_name]["event_prefilter_lowpt_source_particles"].append(int(len(lowpt_particles)))
                lowpt_event_metrics[band_name]["event_fraction_all_hits_lowpt"].append(float(len(lowpt_hits) / max(total_raw_hits, 1)))
                lowpt_event_metrics[band_name]["event_fraction_all_hits_noise"].append(float(len(lowpt_noise_hits_raw) / max(total_raw_hits, 1)))
                lowpt_event_metrics[band_name]["event_prefilter_lowpt_hits_kept"].append(int(np.sum(lowpt_hits["filter_status"].to_numpy(dtype=str) == "kept")) if not lowpt_hits.empty else 0)
                lowpt_event_metrics[band_name]["event_prefilter_lowpt_hits_removed"].append(int(np.sum(lowpt_hits["filter_status"].to_numpy(dtype=str) == "removed")) if not lowpt_hits.empty else 0)
                lowpt_event_metrics[band_name]["event_prefilter_noise_hits_kept"].append(int(np.sum(lowpt_noise_hits_raw["filter_status"].to_numpy(dtype=str) == "kept")) if not lowpt_noise_hits_raw.empty else 0)
                lowpt_event_metrics[band_name]["event_prefilter_noise_hits_removed"].append(int(np.sum(lowpt_noise_hits_raw["filter_status"].to_numpy(dtype=str) == "removed")) if not lowpt_noise_hits_raw.empty else 0)
                lowpt_event_metrics[band_name]["event_prefilter_lowpt_near_valid_hits"].append(lowpt_near_valid_hits)
                lowpt_event_metrics[band_name]["event_prefilter_lowpt_isolated_hits"].append(lowpt_isolated_hits)
                lowpt_event_metrics[band_name]["event_lowpt_hit_near_valid_fraction"].append(float(lowpt_near_valid_hits / max(len(lowpt_hits), 1)))
                lowpt_event_metrics[band_name]["event_lowpt_hit_kept_fraction"].append(float(lowpt_kept_hits / max(len(lowpt_hits), 1)))
                lowpt_event_metrics[band_name]["event_lowpt_hit_kept_fraction_near_valid"].append(
                    float(np.sum(lowpt_kept_mask & lowpt_near_valid_mask) / max(lowpt_near_valid_hits, 1))
                )
                lowpt_event_metrics[band_name]["event_lowpt_hit_kept_fraction_isolated"].append(
                    float(np.sum(lowpt_kept_mask & lowpt_isolated_mask) / max(lowpt_isolated_hits, 1))
                )
                lowpt_event_metrics[band_name]["event_valid_hit_recall"].append(float(valid_kept_hits / max(total_valid_raw_hits, 1)))
                lowpt_event_metrics[band_name]["event_filtered_contamination"].append(float(total_kept_invalid_raw_hits / max(total_kept_hits, 1)))

                lowpt_totals[band_name]["num_events"] += 1
                lowpt_totals[band_name]["num_prefilter_lowpt_hits"] += int(len(lowpt_hits))
                lowpt_totals[band_name]["num_prefilter_noise_hits"] += int(len(lowpt_noise_hits_raw))
                lowpt_totals[band_name]["num_lowpt_source_particles"] += int(len(lowpt_particles))

                lowpt_event_rows.append(
                    {
                        "split": assignments.split,
                        "event_index": int(assignments.event_indices[event_idx]),
                        "sample_id": int(assignments.sample_ids[event_idx]),
                        "event_name": assignments.event_names[event_idx],
                        "band": band_name,
                        "hits_after_filter": int(assignments.hits[event_idx]),
                        "reconstructable_tracks_post_filter": int(assignments.particles[event_idx]),
                        "event_total_raw_hits": total_raw_hits,
                        "event_total_valid_raw_hits": total_valid_raw_hits,
                        "event_prefilter_lowpt_hits": int(len(lowpt_hits)),
                        "event_prefilter_noise_hits": int(len(lowpt_noise_hits_raw)),
                        "event_prefilter_lowpt_source_particles": int(len(lowpt_particles)),
                        "event_fraction_prefilter_lowpt": float(len(lowpt_hits) / total_prefilter_source_hits),
                        "event_fraction_all_hits_lowpt": float(len(lowpt_hits) / max(total_raw_hits, 1)),
                        "event_fraction_all_hits_noise": float(len(lowpt_noise_hits_raw) / max(total_raw_hits, 1)),
                        "event_prefilter_lowpt_near_valid_hits": lowpt_near_valid_hits,
                        "event_prefilter_lowpt_isolated_hits": lowpt_isolated_hits,
                        "event_lowpt_hit_near_valid_fraction": float(lowpt_near_valid_hits / max(len(lowpt_hits), 1)),
                        "event_lowpt_hit_kept_fraction": float(lowpt_kept_hits / max(len(lowpt_hits), 1)),
                        "event_lowpt_hit_kept_fraction_near_valid": float(np.sum(lowpt_kept_mask & lowpt_near_valid_mask) / max(lowpt_near_valid_hits, 1)),
                        "event_lowpt_hit_kept_fraction_isolated": float(np.sum(lowpt_kept_mask & lowpt_isolated_mask) / max(lowpt_isolated_hits, 1)),
                        "event_valid_hit_recall": float(valid_kept_hits / max(total_valid_raw_hits, 1)),
                        "event_filtered_contamination": float(total_kept_invalid_raw_hits / max(total_kept_hits, 1)),
                    }
                )

                region_frame = pd.DataFrame(
                    {
                        "volume_id": hits_scored["volume_id"].to_numpy(dtype=np.int32, copy=False),
                        "layer_id": hits_scored["layer_id"].to_numpy(dtype=np.int32, copy=False),
                        "is_lowpt": np.isin(hit_particle_ids, lowpt_particle_ids),
                        "is_noise": (hit_particle_ids == 0),
                        "is_valid": raw_valid_particle_mask,
                        "is_kept": keep_hits,
                    }
                )
                for (volume_id, layer_id), group in region_frame.groupby(["volume_id", "layer_id"], sort=False):
                    key = (band_name, int(volume_id), int(layer_id))
                    acc = lowpt_region_accumulators.setdefault(
                        key,
                        {
                            "lowpt_hits": 0.0,
                            "kept_lowpt_hits": 0.0,
                            "valid_hits": 0.0,
                            "noise_hits": 0.0,
                            "total_hits": 0.0,
                        },
                    )
                    lowpt_group_mask = group["is_lowpt"].to_numpy(dtype=bool)
                    valid_group_mask = group["is_valid"].to_numpy(dtype=bool)
                    noise_group_mask = group["is_noise"].to_numpy(dtype=bool)
                    kept_group_mask = group["is_kept"].to_numpy(dtype=bool)
                    acc["lowpt_hits"] += float(np.sum(lowpt_group_mask))
                    acc["kept_lowpt_hits"] += float(np.sum(lowpt_group_mask & kept_group_mask))
                    acc["valid_hits"] += float(np.sum(valid_group_mask))
                    acc["noise_hits"] += float(np.sum(noise_group_mask))
                    acc["total_hits"] += float(len(group))

            if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
                print(f"[{assignments.split}] raw-event bundle processed {event_idx + 1}/{len(assignments.band_codes)}")
    finally:
        if hit_eval_file is not None:
            hit_eval_file.close()

    if collect_invalid_source:
        invalid_source_samples = BandInvalidSourceSamples(
            split=assignments.split,
            nonreco_hit_samples=invalid_nonreco_hit_samples,
            nonreco_particle_samples=invalid_nonreco_particle_samples,
            noise_hit_samples=invalid_noise_hit_samples,
            event_metrics={
                band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
                for band_name, metrics in invalid_event_metrics.items()
            },
            totals=invalid_totals,
            failure_summary=pd.DataFrame(invalid_failure_rows).groupby(["split", "band", "primary_fail_reason"], as_index=False).sum(),
        )

    if collect_prefilter_lowpt_noise:
        lowpt_region_rows: list[dict[str, float | int | str]] = []
        for (band_name, volume_id, layer_id), acc in lowpt_region_accumulators.items():
            lowpt_region_rows.append(
                {
                    "split": assignments.split,
                    "band": band_name,
                    "volume_id": volume_id,
                    "layer_id": layer_id,
                    "lowpt_hits": acc["lowpt_hits"],
                    "kept_lowpt_hits": acc["kept_lowpt_hits"],
                    "valid_hits": acc["valid_hits"],
                    "noise_hits": acc["noise_hits"],
                    "total_hits": acc["total_hits"],
                    "lowpt_to_valid_ratio": float(acc["lowpt_hits"] / max(acc["valid_hits"], 1.0)),
                    "lowpt_hit_fraction": float(acc["lowpt_hits"] / max(acc["total_hits"], 1.0)),
                    "noise_hit_fraction": float(acc["noise_hits"] / max(acc["total_hits"], 1.0)),
                    "lowpt_kept_fraction": float(acc["kept_lowpt_hits"] / max(acc["lowpt_hits"], 1.0)),
                }
            )
        prefilter_lowpt_noise_samples = BandPrefilterLowPtNoiseSamples(
            split=assignments.split,
            lowpt_hit_samples=lowpt_hit_samples,
            lowpt_particle_samples=lowpt_particle_samples,
            noise_hit_samples=lowpt_noise_hit_samples,
            event_metrics={
                band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
                for band_name, metrics in lowpt_event_metrics.items()
            },
            totals=lowpt_totals,
            event_diagnostics=pd.DataFrame(lowpt_event_rows),
            region_summary=pd.DataFrame(lowpt_region_rows),
        )

    if collect_prefilter_truth_nonreco:
        for band_name in BAND_NAMES:
            for key in list(truth_nonreco_event_metrics[band_name].keys()):
                truth_nonreco_event_metrics[band_name][key] = np.array(truth_nonreco_event_metrics[band_name][key], dtype=np.float64)
            for key in list(truth_nonreco_noise_event_metrics[band_name].keys()):
                truth_nonreco_noise_event_metrics[band_name][key] = np.array(truth_nonreco_noise_event_metrics[band_name][key], dtype=np.float64)
        prefilter_truth_nonreco_samples = BandPrefilterTruthNonrecoSamples(
            split=assignments.split,
            particle_samples=truth_nonreco_particle_samples,
            event_metrics=truth_nonreco_event_metrics,
            totals=truth_nonreco_totals,
            noise_hit_samples=truth_nonreco_noise_hit_samples,
            noise_event_metrics=truth_nonreco_noise_event_metrics,
        )

    return invalid_source_samples, prefilter_lowpt_noise_samples, prefilter_truth_nonreco_samples


def _collect_band_invalid_source_samples(
    cfg: dict,
    assignments: BandAssignments,
    *,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    rng_seed: int,
    crowding_distance: float,
) -> BandInvalidSourceSamples:
    import h5py

    if assignments.hit_eval_path is None:
        raise RuntimeError(f"Invalid-source analysis requires hit filtering, but split '{assignments.split}' has no hit_eval_path.")

    data_cfg = cfg["data"]
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    min_pt = float(data_cfg["particle_min_pt"])
    max_abs_eta = float(data_cfg["particle_max_abs_eta"])
    min_num_hits = int(data_cfg["particle_min_num_hits"])
    hit_threshold = float(data_cfg.get("hit_filter_threshold", 0.1))
    rng = np.random.default_rng(rng_seed + 3000)

    nonreco_hit_samples = {
        band_name: pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "r",
                "eta",
                "phi",
                "score",
                "min_distance_to_true_same_layer",
                "parent_pt",
                "parent_eta",
                "parent_phi",
                "parent_num_hits_pre",
                "parent_num_hits_post",
                "parent_num_kept_invalid_hits",
                "parent_fail_reason",
                "parent_primary_fail_reason",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    nonreco_particle_samples = {
        band_name: pd.DataFrame(
            columns=[
                "pt",
                "eta",
                "phi",
                "num_hits_pre",
                "num_hits_post",
                "num_kept_invalid_hits",
                "fail_pt",
                "fail_eta",
                "fail_min_hits",
                "fail_reason",
                "primary_fail_reason",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    noise_hit_samples = {
        band_name: pd.DataFrame(
            columns=["x", "y", "z", "r", "eta", "phi", "score", "min_distance_to_true_same_layer", "volume_id", "layer_id", "_sample_key"]
        )
        for band_name in BAND_NAMES
    }
    event_metrics = {
        band_name: {
            "event_kept_nonreco_hits": [],
            "event_kept_noise_hits": [],
            "event_nonreco_source_particles": [],
            "event_fraction_invalid_nonreco": [],
            "event_fraction_invalid_noise": [],
            "event_kept_nonreco_hits_fail_pt": [],
            "event_kept_nonreco_hits_fail_eta": [],
            "event_kept_nonreco_hits_fail_min_hits": [],
        }
        for band_name in BAND_NAMES
    }
    totals = {
        band_name: {
            "num_events": 0,
            "num_kept_nonreco_hits": 0,
            "num_kept_noise_hits": 0,
            "num_nonreco_source_particles": 0,
        }
        for band_name in BAND_NAMES
    }
    failure_rows: list[dict[str, float | int | str]] = []

    print(f"[{assignments.split}] collecting invalid-source samples from raw events")
    with h5py.File(assignments.hit_eval_path, "r") as hit_eval_file:
        for event_idx, band_code in enumerate(assignments.band_codes):
            band_name = BAND_NAMES[int(band_code)]
            hits_raw, particles_raw = _load_raw_trackml_event(assignments.split_dir, assignments.event_names[event_idx], hit_volume_ids)
            hit_scores = _read_hit_filter_scores(hit_eval_file, int(assignments.sample_ids[event_idx]))
            if hit_scores is None or len(hit_scores) != len(hits_raw):
                raise RuntimeError(
                    f"Missing or misaligned hit-filter scores for invalid-source analysis in split '{assignments.split}', sample_id={assignments.sample_ids[event_idx]}."
                )

            keep_hits = np.asarray(hit_scores, dtype=np.float64) >= hit_threshold
            hits_kept = hits_raw.loc[keep_hits, :].copy()
            hits_kept["score"] = np.asarray(hit_scores, dtype=np.float64)[keep_hits]

            particles_all = particles_raw.copy()
            hit_counts_pre = hits_raw["particle_id"].value_counts()
            hit_counts_post = hits_kept["particle_id"].value_counts()
            particles_all["num_hits_pre"] = particles_all["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)
            particles_all["num_hits_post"] = particles_all["particle_id"].map(hit_counts_post).fillna(0).astype(np.int32)

            fail_pt = particles_all["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt
            fail_eta = np.abs(particles_all["eta"].to_numpy(dtype=np.float64, copy=False)) >= max_abs_eta
            fail_min_hits = particles_all["num_hits_post"].to_numpy(dtype=np.int32, copy=False) < min_num_hits
            fail_reason, primary_fail_reason = _compose_fail_reason_labels(fail_pt, fail_eta, fail_min_hits)
            particles_all["fail_pt"] = fail_pt
            particles_all["fail_eta"] = fail_eta
            particles_all["fail_min_hits"] = fail_min_hits
            particles_all["fail_reason"] = fail_reason
            particles_all["primary_fail_reason"] = primary_fail_reason

            valid_particle_ids = particles_all.loc[~(fail_pt | fail_eta | fail_min_hits), "particle_id"].to_numpy(dtype=np.int64, copy=False)
            hits_kept["on_rebuilt_valid_particle"] = hits_kept["particle_id"].isin(valid_particle_ids)
            kept_invalid_hits = hits_kept.loc[~hits_kept["on_rebuilt_valid_particle"], :].copy()
            kept_nonreco_hits = kept_invalid_hits.loc[kept_invalid_hits["particle_id"] != 0, :].copy()
            kept_noise_hits = kept_invalid_hits.loc[kept_invalid_hits["particle_id"] == 0, :].copy()

            valid_hits_kept = hits_kept.loc[hits_kept["on_rebuilt_valid_particle"], :]
            nonreco_min_dist = _same_layer_min_rphiz_distances(kept_nonreco_hits, valid_hits_kept)
            noise_min_dist = _same_layer_min_rphiz_distances(kept_noise_hits, valid_hits_kept)
            kept_nonreco_hits["min_distance_to_true_same_layer"] = nonreco_min_dist
            kept_noise_hits["min_distance_to_true_same_layer"] = noise_min_dist

            nonreco_particles = particles_all.loc[particles_all["particle_id"].isin(kept_nonreco_hits["particle_id"].unique()), :].copy()
            kept_nonreco_counts = kept_nonreco_hits["particle_id"].value_counts()
            nonreco_particles["num_kept_invalid_hits"] = nonreco_particles["particle_id"].map(kept_nonreco_counts).fillna(0).astype(np.int32)

            parent_columns = [
                "parent_pt",
                "parent_eta",
                "parent_phi",
                "parent_num_hits_pre",
                "parent_num_hits_post",
                "parent_num_kept_invalid_hits",
                "parent_fail_reason",
                "parent_primary_fail_reason",
            ]
            if not kept_nonreco_hits.empty:
                parent_lookup = nonreco_particles.set_index("particle_id")[
                    [
                        "pt",
                        "eta",
                        "phi",
                        "num_hits_pre",
                        "num_hits_post",
                        "num_kept_invalid_hits",
                        "fail_reason",
                        "primary_fail_reason",
                    ]
                ].rename(
                    columns={
                        "pt": "parent_pt",
                        "eta": "parent_eta",
                        "phi": "parent_phi",
                        "num_hits_pre": "parent_num_hits_pre",
                        "num_hits_post": "parent_num_hits_post",
                        "num_kept_invalid_hits": "parent_num_kept_invalid_hits",
                        "fail_reason": "parent_fail_reason",
                        "primary_fail_reason": "parent_primary_fail_reason",
                    }
                )
                kept_nonreco_hits = kept_nonreco_hits.join(parent_lookup, on="particle_id")
            for column in parent_columns:
                if column not in kept_nonreco_hits.columns:
                    kept_nonreco_hits[column] = np.nan

            nonreco_hit_frame = kept_nonreco_hits.loc[
                :,
                [
                    "x",
                    "y",
                    "z",
                    "r",
                    "eta",
                    "phi",
                    "score",
                    "min_distance_to_true_same_layer",
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_hits_post",
                    "parent_num_kept_invalid_hits",
                    "parent_fail_reason",
                    "parent_primary_fail_reason",
                ],
            ]
            nonreco_particle_frame = nonreco_particles.loc[
                :,
                [
                    "pt",
                    "eta",
                    "phi",
                    "num_hits_pre",
                    "num_hits_post",
                    "num_kept_invalid_hits",
                    "fail_pt",
                    "fail_eta",
                    "fail_min_hits",
                    "fail_reason",
                    "primary_fail_reason",
                ],
            ]
            noise_hit_frame = kept_noise_hits.loc[
                :,
                ["x", "y", "z", "r", "eta", "phi", "score", "min_distance_to_true_same_layer", "volume_id", "layer_id"],
            ]

            nonreco_hit_samples[band_name] = _priority_sample(nonreco_hit_samples[band_name], nonreco_hit_frame, max_hit_samples_per_band, rng)
            nonreco_particle_samples[band_name] = _priority_sample(
                nonreco_particle_samples[band_name],
                nonreco_particle_frame,
                max_particle_samples_per_band,
                rng,
            )
            noise_hit_samples[band_name] = _priority_sample(noise_hit_samples[band_name], noise_hit_frame, max_hit_samples_per_band, rng)

            num_invalid = max(len(kept_invalid_hits), 1)
            event_metrics[band_name]["event_kept_nonreco_hits"].append(int(len(kept_nonreco_hits)))
            event_metrics[band_name]["event_kept_noise_hits"].append(int(len(kept_noise_hits)))
            event_metrics[band_name]["event_nonreco_source_particles"].append(int(len(nonreco_particles)))
            event_metrics[band_name]["event_fraction_invalid_nonreco"].append(float(len(kept_nonreco_hits) / num_invalid))
            event_metrics[band_name]["event_fraction_invalid_noise"].append(float(len(kept_noise_hits) / num_invalid))
            event_metrics[band_name]["event_kept_nonreco_hits_fail_pt"].append(int(np.sum(kept_nonreco_hits["parent_pt"].to_numpy(dtype=np.float64) <= min_pt)) if not kept_nonreco_hits.empty else 0)
            event_metrics[band_name]["event_kept_nonreco_hits_fail_eta"].append(int(np.sum(np.abs(kept_nonreco_hits["parent_eta"].to_numpy(dtype=np.float64)) >= max_abs_eta)) if not kept_nonreco_hits.empty else 0)
            event_metrics[band_name]["event_kept_nonreco_hits_fail_min_hits"].append(int(np.sum(kept_nonreco_hits["parent_num_hits_post"].to_numpy(dtype=np.int32) < min_num_hits)) if not kept_nonreco_hits.empty else 0)

            totals[band_name]["num_events"] += 1
            totals[band_name]["num_kept_nonreco_hits"] += int(len(kept_nonreco_hits))
            totals[band_name]["num_kept_noise_hits"] += int(len(kept_noise_hits))
            totals[band_name]["num_nonreco_source_particles"] += int(len(nonreco_particles))

            for fail_label in sorted(nonreco_particles["primary_fail_reason"].astype(str).value_counts().index):
                part_count = int(np.sum(nonreco_particles["primary_fail_reason"].astype(str).to_numpy() == fail_label))
                hit_count = int(np.sum(kept_nonreco_hits["parent_primary_fail_reason"].astype(str).to_numpy() == fail_label)) if not kept_nonreco_hits.empty else 0
                failure_rows.append(
                    {
                        "split": assignments.split,
                        "band": band_name,
                        "primary_fail_reason": fail_label,
                        "num_particles": part_count,
                        "num_hits": hit_count,
                    }
                )

            if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
                print(f"[{assignments.split}] invalid-source events processed {event_idx + 1}/{len(assignments.band_codes)}")

    return BandInvalidSourceSamples(
        split=assignments.split,
        nonreco_hit_samples=nonreco_hit_samples,
        nonreco_particle_samples=nonreco_particle_samples,
        noise_hit_samples=noise_hit_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in event_metrics.items()
        },
        totals=totals,
        failure_summary=pd.DataFrame(failure_rows).groupby(["split", "band", "primary_fail_reason"], as_index=False).sum(),
    )


def _invalid_source_feature_map(samples: BandInvalidSourceSamples, band_name: str) -> dict[str, np.ndarray]:
    nonreco_hit_df = samples.nonreco_hit_samples[band_name]
    nonreco_particle_df = samples.nonreco_particle_samples[band_name]
    noise_hit_df = samples.noise_hit_samples[band_name]
    metrics = samples.event_metrics[band_name]
    return {
        "event_kept_nonreco_hits": metrics["event_kept_nonreco_hits"],
        "event_kept_noise_hits": metrics["event_kept_noise_hits"],
        "event_nonreco_source_particles": metrics["event_nonreco_source_particles"],
        "event_fraction_invalid_nonreco": metrics["event_fraction_invalid_nonreco"],
        "event_fraction_invalid_noise": metrics["event_fraction_invalid_noise"],
        "nonreco_particle_pt": nonreco_particle_df["pt"].to_numpy(dtype=np.float64) if "pt" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_particle_eta": nonreco_particle_df["eta"].to_numpy(dtype=np.float64) if "eta" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_particle_phi": nonreco_particle_df["phi"].to_numpy(dtype=np.float64) if "phi" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_particle_num_hits_pre": nonreco_particle_df["num_hits_pre"].to_numpy(dtype=np.float64) if "num_hits_pre" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_particle_num_hits_post": nonreco_particle_df["num_hits_post"].to_numpy(dtype=np.float64) if "num_hits_post" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_particle_num_kept_invalid_hits": nonreco_particle_df["num_kept_invalid_hits"].to_numpy(dtype=np.float64) if "num_kept_invalid_hits" in nonreco_particle_df else np.array([], dtype=np.float64),
        "nonreco_hit_r": nonreco_hit_df["r"].to_numpy(dtype=np.float64) if "r" in nonreco_hit_df else np.array([], dtype=np.float64),
        "nonreco_hit_eta": nonreco_hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in nonreco_hit_df else np.array([], dtype=np.float64),
        "nonreco_hit_phi": nonreco_hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in nonreco_hit_df else np.array([], dtype=np.float64),
        "nonreco_hit_score": nonreco_hit_df["score"].to_numpy(dtype=np.float64) if "score" in nonreco_hit_df else np.array([], dtype=np.float64),
        "nonreco_hit_min_distance_to_true": nonreco_hit_df["min_distance_to_true_same_layer"].to_numpy(dtype=np.float64) if "min_distance_to_true_same_layer" in nonreco_hit_df else np.array([], dtype=np.float64),
        "noise_hit_r": noise_hit_df["r"].to_numpy(dtype=np.float64) if "r" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_eta": noise_hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_phi": noise_hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_score": noise_hit_df["score"].to_numpy(dtype=np.float64) if "score" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_min_distance_to_true": noise_hit_df["min_distance_to_true_same_layer"].to_numpy(dtype=np.float64) if "min_distance_to_true_same_layer" in noise_hit_df else np.array([], dtype=np.float64),
    }


def _write_invalid_source_summary_csv(out_path: Path, samples: BandInvalidSourceSamples) -> None:
    feature_labels = {
        "event_kept_nonreco_hits": "Kept invalid hits on nonreconstructable particles / event",
        "event_kept_noise_hits": "Kept invalid hits on pure noise / event",
        "event_nonreco_source_particles": "Nonreconstructable source particles / event",
        "event_fraction_invalid_nonreco": "Fraction of kept invalid hits on nonreconstructable particles",
        "event_fraction_invalid_noise": "Fraction of kept invalid hits on pure noise",
        "nonreco_particle_pt": "Nonreconstructable particle pT [GeV]",
        "nonreco_particle_eta": "Nonreconstructable particle eta",
        "nonreco_particle_phi": "Nonreconstructable particle phi [rad]",
        "nonreco_particle_num_hits_pre": "Nonreconstructable particle hits before filter",
        "nonreco_particle_num_hits_post": "Nonreconstructable particle hits after filter",
        "nonreco_particle_num_kept_invalid_hits": "Kept invalid hits per nonreconstructable particle",
        "nonreco_hit_r": "Kept nonreconstructable-hit r [m]",
        "nonreco_hit_eta": "Kept nonreconstructable-hit eta",
        "nonreco_hit_phi": "Kept nonreconstructable-hit phi [rad]",
        "nonreco_hit_score": "Kept nonreconstructable-hit score",
        "nonreco_hit_min_distance_to_true": "Kept nonreconstructable-hit distance to valid hit [m]",
        "noise_hit_r": "Kept noise-hit r [m]",
        "noise_hit_eta": "Kept noise-hit eta",
        "noise_hit_phi": "Kept noise-hit phi [rad]",
        "noise_hit_score": "Kept noise-hit score",
        "noise_hit_min_distance_to_true": "Kept noise-hit distance to valid hit [m]",
    }
    rows: list[list[object]] = []
    for band_name in BAND_NAMES:
        feature_values = _invalid_source_feature_map(samples, band_name)
        totals = samples.totals[band_name]
        for feature_key, values in feature_values.items():
            stats = _feature_stats(values)
            rows.append(
                [
                    samples.split,
                    band_name,
                    feature_key,
                    feature_labels[feature_key],
                    int(values.size),
                    totals["num_events"],
                    totals["num_kept_nonreco_hits"],
                    totals["num_kept_noise_hits"],
                    totals["num_nonreco_source_particles"],
                    stats["min"],
                    stats["p05"],
                    stats["median"],
                    stats["mean"],
                    stats["std"],
                    stats["p95"],
                    stats["max"],
                ]
            )
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "band",
                "feature",
                "label",
                "sample_size",
                "num_events_in_band",
                "num_kept_nonreco_hits_in_band",
                "num_kept_noise_hits_in_band",
                "num_nonreco_source_particles_in_band",
                "min",
                "p05",
                "median",
                "mean",
                "std",
                "p95",
                "max",
            ]
        )
        writer.writerows(rows)


def _plot_invalid_source_comparison(samples: BandInvalidSourceSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_kept_nonreco_hits": "Nonreco invalid hits / event",
        "event_kept_noise_hits": "Noise invalid hits / event",
        "event_nonreco_source_particles": "Nonreco source particles / event",
        "event_fraction_invalid_nonreco": "Invalid-hit fraction from nonreco particles",
        "event_fraction_invalid_noise": "Invalid-hit fraction from noise",
        "nonreco_particle_pt": "Nonreco particle pT [GeV]",
        "nonreco_particle_eta": "Nonreco particle eta",
        "nonreco_particle_phi": "Nonreco particle phi [rad]",
        "nonreco_particle_num_hits_pre": "Nonreco particle hits before filter",
        "nonreco_particle_num_hits_post": "Nonreco particle hits after filter",
        "nonreco_particle_num_kept_invalid_hits": "Kept invalid hits / nonreco particle",
        "nonreco_hit_r": "Kept nonreco-hit r [m]",
        "nonreco_hit_eta": "Kept nonreco-hit eta",
        "nonreco_hit_phi": "Kept nonreco-hit phi [rad]",
        "nonreco_hit_score": "Kept nonreco-hit score",
        "nonreco_hit_min_distance_to_true": "Nonreco-hit distance to valid hit [m]",
    }
    feature_order = list(feature_labels)
    per_band_features = {band_name: _invalid_source_feature_map(samples, band_name) for band_name in BAND_NAMES}

    fig, axes = plt.subplots(4, 4, figsize=(15.6, 13.2), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    axes_flat = axes.ravel()
    legend_handles = []
    legend_labels = []
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        lower_vals = lower_vals[np.isfinite(lower_vals)]
        upper_vals = upper_vals[np.isfinite(upper_vals)]
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key.endswith("_pt"):
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif feature_key.endswith("_phi"):
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        elif "fraction" in feature_key:
            bins, xscale = np.linspace(0.0, 1.0, 51), "linear"
        elif "num_hits" in feature_key or feature_key.endswith("_hits") or feature_key.endswith("_particles") or feature_key.startswith("event_"):
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"
        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=num_events_by_band["lower_band"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=num_events_by_band["upper_band"])
        elif feature_key.startswith("nonreco_particle_"):
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.nonreco_particle_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_nonreco_source_particles"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.nonreco_particle_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_nonreco_source_particles"],
                num_events=num_events_by_band["upper_band"],
            )
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.nonreco_hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_kept_nonreco_hits"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.nonreco_hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_kept_nonreco_hits"],
                num_events=num_events_by_band["upper_band"],
            )
        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label="lower band",
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label="upper band",
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = ["lower band", "upper band"]
        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_event_hist_ylabel(per_band=True) if feature_key.startswith("event_") else _hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    if len(feature_order) < len(axes_flat):
        legend_ax = axes_flat[len(feature_order)]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)
        unused_start = len(feature_order) + 1
    else:
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=2, frameon=False)
        unused_start = len(feature_order)
    for ax in axes_flat[unused_start:]:
        ax.axis("off")
    fig.suptitle(f"{samples.split}: kept invalid hits from nonreconstructable particles", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_noise_hit_comparison(samples: BandInvalidSourceSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_kept_noise_hits": "Noise invalid hits / event",
        "event_fraction_invalid_noise": "Invalid-hit fraction from noise",
        "noise_hit_r": "Noise-hit r [m]",
        "noise_hit_eta": "Noise-hit eta",
        "noise_hit_phi": "Noise-hit phi [rad]",
        "noise_hit_score": "Noise-hit score",
        "noise_hit_min_distance_to_true": "Noise-hit distance to valid hit [m]",
    }
    feature_order = list(feature_labels)
    per_band_features = {band_name: _invalid_source_feature_map(samples, band_name) for band_name in BAND_NAMES}

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.30)
    axes_flat = axes.ravel()
    legend_handles = []
    legend_labels = []
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}
    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key.endswith("_phi"):
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        elif "fraction" in feature_key:
            bins, xscale = np.linspace(0.0, 1.0, 51), "linear"
        elif feature_key.endswith("_hits"):
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"
        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=num_events_by_band["lower_band"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=num_events_by_band["upper_band"])
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.noise_hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_kept_noise_hits"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.noise_hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_kept_noise_hits"],
                num_events=num_events_by_band["upper_band"],
            )
        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label="lower band",
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label="upper band",
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = ["lower band", "upper band"]
        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    if len(feature_order) < len(axes_flat):
        legend_ax = axes_flat[len(feature_order)]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)
        unused_start = len(feature_order) + 1
    else:
        fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=2, frameon=False)
        unused_start = len(feature_order)
    for ax in axes_flat[unused_start:]:
        ax.axis("off")
    fig.suptitle(f"{samples.split}: kept invalid hits from pure noise", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_nonreco_failure_breakdown(samples: BandInvalidSourceSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    df = samples.failure_summary.copy()
    if df.empty:
        return

    reason_order = ["pt", "eta", "min_hits", "multiple"]
    particle_pivot = df.pivot_table(index="primary_fail_reason", columns="band", values="num_particles", aggfunc="sum").fillna(0.0)
    hit_pivot = df.pivot_table(index="primary_fail_reason", columns="band", values="num_hits", aggfunc="sum").fillna(0.0)
    for pivot in (particle_pivot, hit_pivot):
        for band_name in BAND_NAMES:
            if band_name not in pivot.columns:
                pivot[band_name] = 0.0
    plot_reasons = [reason for reason in reason_order if reason in particle_pivot.index or reason in hit_pivot.index]
    x = np.arange(len(plot_reasons), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=False)
    fig.subplots_adjust(top=0.86, wspace=0.28)
    for ax, pivot, title, ylabel in (
        (axes[0], particle_pivot, "Source nonreconstructable particles by primary fail reason", "Particles"),
        (axes[1], hit_pivot, "Kept invalid hits by source fail reason", "Hits"),
    ):
        lower_vals = np.array([pivot.loc[reason, BAND_NAMES[0]] if reason in pivot.index else 0.0 for reason in plot_reasons], dtype=np.float64)
        upper_vals = np.array([pivot.loc[reason, BAND_NAMES[1]] if reason in pivot.index else 0.0 for reason in plot_reasons], dtype=np.float64)
        if _hist_count_per_event():
            lower_vals /= max(samples.totals[BAND_NAMES[0]]["num_events"], 1)
            upper_vals /= max(samples.totals[BAND_NAMES[1]]["num_events"], 1)
        ax.bar(x - width / 2, lower_vals, width=width, color=BAND_COLORS[BAND_NAMES[0]], label=BAND_NAMES[0].replace("_", " "), alpha=0.85)
        ax.bar(x + width / 2, upper_vals, width=width, color=BAND_COLORS[BAND_NAMES[1]], label=BAND_NAMES[1].replace("_", " "), alpha=0.85)
        ax.set_xticks(x, plot_reasons)
        ax.set_ylabel(f"{ylabel} / event" if _hist_count_per_event() else ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="y")
        ax.set_axisbelow(True)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{samples.split}: why kept invalid nonreconstructable sources fail tracking cuts", fontsize=13, y=0.965)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_nonreco_reason_feature_summary_csv(out_path: Path, samples: BandInvalidSourceSamples) -> None:
    rows: list[list[object]] = []
    feature_specs = [
        ("pt", "Source particle pT [GeV]"),
        ("eta", "Source particle eta"),
        ("num_hits_post", "Source particle hits after filter"),
        ("num_kept_invalid_hits", "Kept invalid hits per source particle"),
    ]
    for band_name in BAND_NAMES:
        particle_df = samples.nonreco_particle_samples[band_name]
        for reason in sorted(particle_df["primary_fail_reason"].astype(str).unique()) if not particle_df.empty else []:
            reason_df = particle_df[particle_df["primary_fail_reason"].astype(str) == reason]
            for feature_key, label in feature_specs:
                values = reason_df[feature_key].to_numpy(dtype=np.float64)
                stats = _feature_stats(values)
                rows.append(
                    [
                        samples.split,
                        band_name,
                        reason,
                        feature_key,
                        label,
                        int(len(values)),
                        stats["min"],
                        stats["p05"],
                        stats["median"],
                        stats["mean"],
                        stats["std"],
                        stats["p95"],
                        stats["max"],
                    ]
                )
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "band", "primary_fail_reason", "feature", "label", "sample_size", "min", "p05", "median", "mean", "std", "p95", "max"])
        writer.writerows(rows)


def _plot_nonreco_reason_conditioned_comparison(samples: BandInvalidSourceSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    reason_order = ["pt", "eta", "min_hits"]
    feature_specs = [
        ("pt", "Source particle pT [GeV]"),
        ("eta", "Source particle eta"),
        ("num_hits_post", "Source particle hits after filter"),
        ("num_kept_invalid_hits", "Kept invalid hits / source particle"),
    ]
    fig, axes = plt.subplots(len(reason_order), len(feature_specs), figsize=(15.4, 10.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.30)
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}
    reason_totals = samples.failure_summary.pivot_table(
        index="primary_fail_reason",
        columns="band",
        values="num_particles",
        aggfunc="sum",
    ).fillna(0.0)

    for row_idx, reason in enumerate(reason_order):
        for col_idx, (feature_key, title) in enumerate(feature_specs):
            ax = axes[row_idx, col_idx]
            lower_df = samples.nonreco_particle_samples["lower_band"]
            upper_df = samples.nonreco_particle_samples["upper_band"]
            lower_vals = lower_df.loc[lower_df["primary_fail_reason"].astype(str) == reason, feature_key].to_numpy(dtype=np.float64)
            upper_vals = upper_df.loc[upper_df["primary_fail_reason"].astype(str) == reason, feature_key].to_numpy(dtype=np.float64)
            lower_total = int(reason_totals.loc[reason, "lower_band"]) if reason in reason_totals.index and "lower_band" in reason_totals.columns else 0
            upper_total = int(reason_totals.loc[reason, "upper_band"]) if reason in reason_totals.index and "upper_band" in reason_totals.columns else 0
            combined = _combined_values(lower_vals, upper_vals)
            if feature_key == "pt":
                bins, xscale = _feature_bins_and_scale("particle_pt", combined)
            elif feature_key == "eta":
                bins, xscale = _continuous_bins(combined), "linear"
            else:
                bins, xscale = _count_bins(combined), "linear"
            ax.hist(
                lower_vals,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(lower_vals),
                    sampled_population_size=len(lower_vals),
                    total_population=lower_total,
                    num_events=num_events_by_band["lower_band"],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5,
                color=BAND_COLORS["lower_band"],
                label="lower band",
            )
            ax.hist(
                upper_vals,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(upper_vals),
                    sampled_population_size=len(upper_vals),
                    total_population=upper_total,
                    num_events=num_events_by_band["upper_band"],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5,
                color=BAND_COLORS["upper_band"],
                label="upper band",
            )
            ax.set_title(f"{reason}: {title}", fontsize=10.5)
            ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
            ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
            ax.set_axisbelow(True)
            if xscale == "log":
                ax.set_xscale("log")
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle(f"{samples.split}: source-particle distributions split by nonreconstructable reason", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _collect_band_prefilter_lowpt_noise_samples(
    cfg: dict,
    assignments: BandAssignments,
    *,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    rng_seed: int,
    crowding_distance: float,
) -> BandPrefilterLowPtNoiseSamples:
    import h5py

    if assignments.hit_eval_path is None:
        raise RuntimeError(f"Pre-filter low-pT/noise analysis requires hit filtering, but split '{assignments.split}' has no hit_eval_path.")

    data_cfg = cfg["data"]
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    min_pt = float(data_cfg["particle_min_pt"])
    max_abs_eta = float(data_cfg["particle_max_abs_eta"])
    min_num_hits = int(data_cfg["particle_min_num_hits"])
    hit_threshold = float(data_cfg.get("hit_filter_threshold", 0.1))
    rng = np.random.default_rng(rng_seed + 4000)

    lowpt_hit_samples = {
        band_name: pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "r",
                "eta",
                "phi",
                "score",
                "filter_status",
                "min_distance_to_valid_same_layer",
                "proximity_bucket",
                "volume_id",
                "layer_id",
                "parent_pt",
                "parent_eta",
                "parent_phi",
                "parent_num_hits_pre",
                "parent_num_lowpt_hits_total",
                "parent_num_lowpt_hits_kept",
                "parent_num_lowpt_hits_removed",
                "parent_kept_hit_fraction",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    lowpt_particle_samples = {
        band_name: pd.DataFrame(
            columns=[
                "pt",
                "eta",
                "phi",
                "num_hits_pre",
                "num_lowpt_hits_total",
                "num_lowpt_hits_kept",
                "num_lowpt_hits_removed",
                "kept_hit_fraction",
                "_sample_key",
            ]
        )
        for band_name in BAND_NAMES
    }
    noise_hit_samples = {
        band_name: pd.DataFrame(
            columns=["x", "y", "z", "r", "eta", "phi", "score", "filter_status", "volume_id", "layer_id", "_sample_key"]
        )
        for band_name in BAND_NAMES
    }
    event_metrics = {
        band_name: {
            "event_prefilter_lowpt_hits": [],
            "event_prefilter_noise_hits": [],
            "event_fraction_prefilter_lowpt": [],
            "event_fraction_prefilter_noise": [],
            "event_total_raw_hits": [],
            "event_total_valid_raw_hits": [],
            "event_prefilter_lowpt_source_particles": [],
            "event_fraction_all_hits_lowpt": [],
            "event_fraction_all_hits_noise": [],
            "event_prefilter_lowpt_hits_kept": [],
            "event_prefilter_lowpt_hits_removed": [],
            "event_prefilter_noise_hits_kept": [],
            "event_prefilter_noise_hits_removed": [],
            "event_prefilter_lowpt_near_valid_hits": [],
            "event_prefilter_lowpt_isolated_hits": [],
            "event_lowpt_hit_near_valid_fraction": [],
            "event_lowpt_hit_kept_fraction": [],
            "event_lowpt_hit_kept_fraction_near_valid": [],
            "event_lowpt_hit_kept_fraction_isolated": [],
            "event_valid_hit_recall": [],
            "event_filtered_contamination": [],
        }
        for band_name in BAND_NAMES
    }
    totals = {
        band_name: {
            "num_events": 0,
            "num_prefilter_lowpt_hits": 0,
            "num_prefilter_noise_hits": 0,
            "num_lowpt_source_particles": 0,
        }
        for band_name in BAND_NAMES
    }
    event_rows: list[dict[str, float | int | str]] = []
    region_accumulators: dict[tuple[str, int, int], dict[str, float]] = {}

    print(f"[{assignments.split}] collecting pre-filter low-pT-versus-noise samples from raw events")
    with h5py.File(assignments.hit_eval_path, "r") as hit_eval_file:
        for event_idx, band_code in enumerate(assignments.band_codes):
            band_name = BAND_NAMES[int(band_code)]
            hits_raw, particles_raw = _load_raw_trackml_event(assignments.split_dir, assignments.event_names[event_idx], hit_volume_ids)
            hit_scores = _read_hit_filter_scores(hit_eval_file, int(assignments.sample_ids[event_idx]))
            if hit_scores is None or len(hit_scores) != len(hits_raw):
                raise RuntimeError(
                    f"Missing or misaligned hit-filter scores for pre-filter low-pT/noise analysis in split '{assignments.split}', sample_id={assignments.sample_ids[event_idx]}."
                )

            hits_raw = hits_raw.copy()
            hits_raw["score"] = np.asarray(hit_scores, dtype=np.float64)
            keep_hits = hits_raw["score"].to_numpy(dtype=np.float64, copy=False) >= float(data_cfg.get("hit_filter_threshold", 0.1))
            hits_raw["filter_status"] = np.where(keep_hits, "kept", "removed")

            particles_all = particles_raw.copy()
            hit_counts_pre = hits_raw["particle_id"].value_counts()
            particles_all["num_hits_pre"] = particles_all["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)
            valid_particle_mask_pre = (
                (particles_all["pt"].to_numpy(dtype=np.float64, copy=False) > min_pt)
                & (np.abs(particles_all["eta"].to_numpy(dtype=np.float64, copy=False)) < max_abs_eta)
                & (particles_all["num_hits_pre"].to_numpy(dtype=np.int32, copy=False) >= min_num_hits)
            )
            valid_particle_ids_pre = particles_all.loc[valid_particle_mask_pre, "particle_id"].to_numpy(dtype=np.int64, copy=False)
            valid_hits_raw = hits_raw.loc[hits_raw["particle_id"].isin(valid_particle_ids_pre), :].copy()
            invalid_hits_raw = hits_raw.loc[~hits_raw["particle_id"].isin(valid_particle_ids_pre), :].copy()

            lowpt_particle_ids = particles_all.loc[particles_all["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt, "particle_id"].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            lowpt_hits = hits_raw.loc[hits_raw["particle_id"].isin(lowpt_particle_ids), :].copy()
            noise_hits = hits_raw.loc[hits_raw["particle_id"] == 0, :].copy()
            lowpt_min_distances = _same_layer_min_rphiz_distances(lowpt_hits, valid_hits_raw)
            lowpt_hits["min_distance_to_valid_same_layer"] = lowpt_min_distances
            lowpt_hits["proximity_bucket"] = np.where(
                np.isfinite(lowpt_min_distances) & (lowpt_min_distances < crowding_distance),
                "near_valid",
                "isolated",
            )

            lowpt_particles = particles_all.loc[particles_all["particle_id"].isin(lowpt_hits["particle_id"].unique()), :].copy()
            lowpt_hit_counts_total = lowpt_hits["particle_id"].value_counts()
            lowpt_hit_counts_kept = lowpt_hits.loc[lowpt_hits["filter_status"] == "kept", "particle_id"].value_counts()
            lowpt_particles["num_lowpt_hits_total"] = lowpt_particles["particle_id"].map(lowpt_hit_counts_total).fillna(0).astype(np.int32)
            lowpt_particles["num_lowpt_hits_kept"] = lowpt_particles["particle_id"].map(lowpt_hit_counts_kept).fillna(0).astype(np.int32)
            lowpt_particles["num_lowpt_hits_removed"] = (
                lowpt_particles["num_lowpt_hits_total"].to_numpy(dtype=np.int32, copy=False)
                - lowpt_particles["num_lowpt_hits_kept"].to_numpy(dtype=np.int32, copy=False)
            )
            lowpt_particles["kept_hit_fraction"] = np.divide(
                lowpt_particles["num_lowpt_hits_kept"].to_numpy(dtype=np.float64, copy=False),
                np.maximum(lowpt_particles["num_lowpt_hits_total"].to_numpy(dtype=np.float64, copy=False), 1.0),
            )

            parent_columns = [
                "parent_pt",
                "parent_eta",
                "parent_phi",
                "parent_num_hits_pre",
                "parent_num_lowpt_hits_total",
                "parent_num_lowpt_hits_kept",
                "parent_num_lowpt_hits_removed",
                "parent_kept_hit_fraction",
            ]
            if not lowpt_hits.empty:
                parent_lookup = lowpt_particles.set_index("particle_id")[
                    [
                        "pt",
                        "eta",
                        "phi",
                        "num_hits_pre",
                        "num_lowpt_hits_total",
                        "num_lowpt_hits_kept",
                        "num_lowpt_hits_removed",
                        "kept_hit_fraction",
                    ]
                ].rename(
                    columns={
                        "pt": "parent_pt",
                        "eta": "parent_eta",
                        "phi": "parent_phi",
                        "num_hits_pre": "parent_num_hits_pre",
                        "num_lowpt_hits_total": "parent_num_lowpt_hits_total",
                        "num_lowpt_hits_kept": "parent_num_lowpt_hits_kept",
                        "num_lowpt_hits_removed": "parent_num_lowpt_hits_removed",
                        "kept_hit_fraction": "parent_kept_hit_fraction",
                    }
                )
                lowpt_hits = lowpt_hits.join(parent_lookup, on="particle_id")
            for column in parent_columns:
                if column not in lowpt_hits.columns:
                    lowpt_hits[column] = np.nan

            lowpt_hit_frame = lowpt_hits.loc[
                :,
                [
                    "x",
                    "y",
                    "z",
                    "r",
                    "eta",
                    "phi",
                    "score",
                    "filter_status",
                    "min_distance_to_valid_same_layer",
                    "proximity_bucket",
                    "volume_id",
                    "layer_id",
                    "parent_pt",
                    "parent_eta",
                    "parent_phi",
                    "parent_num_hits_pre",
                    "parent_num_lowpt_hits_total",
                    "parent_num_lowpt_hits_kept",
                    "parent_num_lowpt_hits_removed",
                    "parent_kept_hit_fraction",
                ],
            ]
            lowpt_particle_frame = lowpt_particles.loc[
                :,
                [
                    "pt",
                    "eta",
                    "phi",
                    "num_hits_pre",
                    "num_lowpt_hits_total",
                    "num_lowpt_hits_kept",
                    "num_lowpt_hits_removed",
                    "kept_hit_fraction",
                ],
            ]
            noise_hit_frame = noise_hits.loc[:, ["x", "y", "z", "r", "eta", "phi", "score", "filter_status", "volume_id", "layer_id"]]

            lowpt_hit_samples[band_name] = _priority_sample(lowpt_hit_samples[band_name], lowpt_hit_frame, max_hit_samples_per_band, rng)
            lowpt_particle_samples[band_name] = _priority_sample(
                lowpt_particle_samples[band_name],
                lowpt_particle_frame,
                max_particle_samples_per_band,
                rng,
            )
            noise_hit_samples[band_name] = _priority_sample(noise_hit_samples[band_name], noise_hit_frame, max_hit_samples_per_band, rng)

            total_prefilter_source_hits = max(int(len(lowpt_hits) + len(noise_hits)), 1)
            total_raw_hits = int(len(hits_raw))
            total_valid_raw_hits = int(len(valid_hits_raw))
            total_kept_hits = int(np.sum(keep_hits))
            total_kept_invalid_raw_hits = int(np.sum((~hits_raw["particle_id"].isin(valid_particle_ids_pre)).to_numpy(dtype=bool) & keep_hits))
            lowpt_near_valid_mask = lowpt_hits["proximity_bucket"].to_numpy(dtype=str) == "near_valid" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
            lowpt_kept_mask = lowpt_hits["filter_status"].to_numpy(dtype=str) == "kept" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
            lowpt_isolated_mask = lowpt_hits["proximity_bucket"].to_numpy(dtype=str) == "isolated" if not lowpt_hits.empty else np.zeros(0, dtype=bool)
            lowpt_near_valid_hits = int(np.sum(lowpt_near_valid_mask))
            lowpt_isolated_hits = int(np.sum(lowpt_isolated_mask))
            lowpt_kept_hits = int(np.sum(lowpt_kept_mask))
            valid_kept_hits = int(np.sum(hits_raw["particle_id"].isin(valid_particle_ids_pre).to_numpy(dtype=bool) & keep_hits))
            event_metrics[band_name]["event_prefilter_lowpt_hits"].append(int(len(lowpt_hits)))
            event_metrics[band_name]["event_prefilter_noise_hits"].append(int(len(noise_hits)))
            event_metrics[band_name]["event_fraction_prefilter_lowpt"].append(float(len(lowpt_hits) / total_prefilter_source_hits))
            event_metrics[band_name]["event_fraction_prefilter_noise"].append(float(len(noise_hits) / total_prefilter_source_hits))
            event_metrics[band_name]["event_total_raw_hits"].append(total_raw_hits)
            event_metrics[band_name]["event_total_valid_raw_hits"].append(total_valid_raw_hits)
            event_metrics[band_name]["event_prefilter_lowpt_source_particles"].append(int(len(lowpt_particles)))
            event_metrics[band_name]["event_fraction_all_hits_lowpt"].append(float(len(lowpt_hits) / max(total_raw_hits, 1)))
            event_metrics[band_name]["event_fraction_all_hits_noise"].append(float(len(noise_hits) / max(total_raw_hits, 1)))
            event_metrics[band_name]["event_prefilter_lowpt_hits_kept"].append(int(np.sum(lowpt_hits["filter_status"].to_numpy(dtype=str) == "kept")) if not lowpt_hits.empty else 0)
            event_metrics[band_name]["event_prefilter_lowpt_hits_removed"].append(int(np.sum(lowpt_hits["filter_status"].to_numpy(dtype=str) == "removed")) if not lowpt_hits.empty else 0)
            event_metrics[band_name]["event_prefilter_noise_hits_kept"].append(int(np.sum(noise_hits["filter_status"].to_numpy(dtype=str) == "kept")) if not noise_hits.empty else 0)
            event_metrics[band_name]["event_prefilter_noise_hits_removed"].append(int(np.sum(noise_hits["filter_status"].to_numpy(dtype=str) == "removed")) if not noise_hits.empty else 0)
            event_metrics[band_name]["event_prefilter_lowpt_near_valid_hits"].append(lowpt_near_valid_hits)
            event_metrics[band_name]["event_prefilter_lowpt_isolated_hits"].append(lowpt_isolated_hits)
            event_metrics[band_name]["event_lowpt_hit_near_valid_fraction"].append(float(lowpt_near_valid_hits / max(len(lowpt_hits), 1)))
            event_metrics[band_name]["event_lowpt_hit_kept_fraction"].append(float(lowpt_kept_hits / max(len(lowpt_hits), 1)))
            event_metrics[band_name]["event_lowpt_hit_kept_fraction_near_valid"].append(
                float(np.sum(lowpt_kept_mask & lowpt_near_valid_mask) / max(lowpt_near_valid_hits, 1))
            )
            event_metrics[band_name]["event_lowpt_hit_kept_fraction_isolated"].append(
                float(np.sum(lowpt_kept_mask & lowpt_isolated_mask) / max(lowpt_isolated_hits, 1))
            )
            event_metrics[band_name]["event_valid_hit_recall"].append(float(valid_kept_hits / max(total_valid_raw_hits, 1)))
            event_metrics[band_name]["event_filtered_contamination"].append(float(total_kept_invalid_raw_hits / max(total_kept_hits, 1)))

            totals[band_name]["num_events"] += 1
            totals[band_name]["num_prefilter_lowpt_hits"] += int(len(lowpt_hits))
            totals[band_name]["num_prefilter_noise_hits"] += int(len(noise_hits))
            totals[band_name]["num_lowpt_source_particles"] += int(len(lowpt_particles))

            event_rows.append(
                {
                    "split": assignments.split,
                    "event_index": int(assignments.event_indices[event_idx]),
                    "sample_id": int(assignments.sample_ids[event_idx]),
                    "event_name": assignments.event_names[event_idx],
                    "band": band_name,
                    "hits_after_filter": int(assignments.hits[event_idx]),
                    "reconstructable_tracks_post_filter": int(assignments.particles[event_idx]),
                    "event_total_raw_hits": total_raw_hits,
                    "event_total_valid_raw_hits": total_valid_raw_hits,
                    "event_prefilter_lowpt_hits": int(len(lowpt_hits)),
                    "event_prefilter_noise_hits": int(len(noise_hits)),
                    "event_prefilter_lowpt_source_particles": int(len(lowpt_particles)),
                    "event_fraction_prefilter_lowpt": float(len(lowpt_hits) / total_prefilter_source_hits),
                    "event_fraction_all_hits_lowpt": float(len(lowpt_hits) / max(total_raw_hits, 1)),
                    "event_fraction_all_hits_noise": float(len(noise_hits) / max(total_raw_hits, 1)),
                    "event_prefilter_lowpt_near_valid_hits": lowpt_near_valid_hits,
                    "event_prefilter_lowpt_isolated_hits": lowpt_isolated_hits,
                    "event_lowpt_hit_near_valid_fraction": float(lowpt_near_valid_hits / max(len(lowpt_hits), 1)),
                    "event_lowpt_hit_kept_fraction": float(lowpt_kept_hits / max(len(lowpt_hits), 1)),
                    "event_lowpt_hit_kept_fraction_near_valid": float(np.sum(lowpt_kept_mask & lowpt_near_valid_mask) / max(lowpt_near_valid_hits, 1)),
                    "event_lowpt_hit_kept_fraction_isolated": float(np.sum(lowpt_kept_mask & lowpt_isolated_mask) / max(lowpt_isolated_hits, 1)),
                    "event_valid_hit_recall": float(valid_kept_hits / max(total_valid_raw_hits, 1)),
                    "event_filtered_contamination": float(total_kept_invalid_raw_hits / max(total_kept_hits, 1)),
                }
            )

            region_frame = pd.DataFrame(
                {
                    "volume_id": hits_raw["volume_id"].to_numpy(dtype=np.int32, copy=False),
                    "layer_id": hits_raw["layer_id"].to_numpy(dtype=np.int32, copy=False),
                    "is_lowpt": hits_raw["particle_id"].isin(lowpt_particle_ids).to_numpy(dtype=bool),
                    "is_noise": (hits_raw["particle_id"].to_numpy(dtype=np.int64, copy=False) == 0),
                    "is_valid": hits_raw["particle_id"].isin(valid_particle_ids_pre).to_numpy(dtype=bool),
                    "is_kept": keep_hits,
                }
            )
            for (volume_id, layer_id), group in region_frame.groupby(["volume_id", "layer_id"], sort=False):
                key = (band_name, int(volume_id), int(layer_id))
                acc = region_accumulators.setdefault(
                    key,
                    {
                        "lowpt_hits": 0.0,
                        "kept_lowpt_hits": 0.0,
                        "valid_hits": 0.0,
                        "noise_hits": 0.0,
                        "total_hits": 0.0,
                    },
                )
                lowpt_group_mask = group["is_lowpt"].to_numpy(dtype=bool)
                valid_group_mask = group["is_valid"].to_numpy(dtype=bool)
                noise_group_mask = group["is_noise"].to_numpy(dtype=bool)
                kept_group_mask = group["is_kept"].to_numpy(dtype=bool)
                acc["lowpt_hits"] += float(np.sum(lowpt_group_mask))
                acc["kept_lowpt_hits"] += float(np.sum(lowpt_group_mask & kept_group_mask))
                acc["valid_hits"] += float(np.sum(valid_group_mask))
                acc["noise_hits"] += float(np.sum(noise_group_mask))
                acc["total_hits"] += float(len(group))

            if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
                print(f"[{assignments.split}] pre-filter low-pT/noise events processed {event_idx + 1}/{len(assignments.band_codes)}")

    region_rows: list[dict[str, float | int | str]] = []
    for (band_name, volume_id, layer_id), acc in region_accumulators.items():
        region_rows.append(
            {
                "split": assignments.split,
                "band": band_name,
                "volume_id": volume_id,
                "layer_id": layer_id,
                "lowpt_hits": acc["lowpt_hits"],
                "kept_lowpt_hits": acc["kept_lowpt_hits"],
                "valid_hits": acc["valid_hits"],
                "noise_hits": acc["noise_hits"],
                "total_hits": acc["total_hits"],
                "lowpt_to_valid_ratio": float(acc["lowpt_hits"] / max(acc["valid_hits"], 1.0)),
                "lowpt_hit_fraction": float(acc["lowpt_hits"] / max(acc["total_hits"], 1.0)),
                "noise_hit_fraction": float(acc["noise_hits"] / max(acc["total_hits"], 1.0)),
                "lowpt_kept_fraction": float(acc["kept_lowpt_hits"] / max(acc["lowpt_hits"], 1.0)),
            }
        )

    return BandPrefilterLowPtNoiseSamples(
        split=assignments.split,
        lowpt_hit_samples=lowpt_hit_samples,
        lowpt_particle_samples=lowpt_particle_samples,
        noise_hit_samples=noise_hit_samples,
        event_metrics={
            band_name: {key: np.asarray(values, dtype=np.float64) for key, values in metrics.items()}
            for band_name, metrics in event_metrics.items()
        },
        totals=totals,
        event_diagnostics=pd.DataFrame(event_rows),
        region_summary=pd.DataFrame(region_rows),
    )


def _prefilter_lowpt_noise_feature_map(samples: BandPrefilterLowPtNoiseSamples, band_name: str) -> dict[str, np.ndarray]:
    lowpt_hit_df = samples.lowpt_hit_samples[band_name]
    lowpt_particle_df = samples.lowpt_particle_samples[band_name]
    noise_hit_df = samples.noise_hit_samples[band_name]
    metrics = samples.event_metrics[band_name]
    return {
        "event_prefilter_lowpt_hits": metrics["event_prefilter_lowpt_hits"],
        "event_prefilter_noise_hits": metrics["event_prefilter_noise_hits"],
        "event_fraction_prefilter_lowpt": metrics["event_fraction_prefilter_lowpt"],
        "event_total_raw_hits": metrics["event_total_raw_hits"],
        "event_total_valid_raw_hits": metrics["event_total_valid_raw_hits"],
        "event_prefilter_lowpt_source_particles": metrics["event_prefilter_lowpt_source_particles"],
        "event_fraction_all_hits_lowpt": metrics["event_fraction_all_hits_lowpt"],
        "event_fraction_all_hits_noise": metrics["event_fraction_all_hits_noise"],
        "event_prefilter_lowpt_near_valid_hits": metrics["event_prefilter_lowpt_near_valid_hits"],
        "event_prefilter_lowpt_isolated_hits": metrics["event_prefilter_lowpt_isolated_hits"],
        "event_lowpt_hit_near_valid_fraction": metrics["event_lowpt_hit_near_valid_fraction"],
        "event_lowpt_hit_kept_fraction": metrics["event_lowpt_hit_kept_fraction"],
        "event_lowpt_hit_kept_fraction_near_valid": metrics["event_lowpt_hit_kept_fraction_near_valid"],
        "event_lowpt_hit_kept_fraction_isolated": metrics["event_lowpt_hit_kept_fraction_isolated"],
        "event_valid_hit_recall": metrics["event_valid_hit_recall"],
        "event_filtered_contamination": metrics["event_filtered_contamination"],
        "lowpt_particle_pt": lowpt_particle_df["pt"].to_numpy(dtype=np.float64) if "pt" in lowpt_particle_df else np.array([], dtype=np.float64),
        "lowpt_particle_eta": lowpt_particle_df["eta"].to_numpy(dtype=np.float64) if "eta" in lowpt_particle_df else np.array([], dtype=np.float64),
        "lowpt_particle_num_hits_pre": lowpt_particle_df["num_hits_pre"].to_numpy(dtype=np.float64) if "num_hits_pre" in lowpt_particle_df else np.array([], dtype=np.float64),
        "lowpt_particle_num_lowpt_hits_total": (
            lowpt_particle_df["num_lowpt_hits_total"].to_numpy(dtype=np.float64) if "num_lowpt_hits_total" in lowpt_particle_df else np.array([], dtype=np.float64)
        ),
        "lowpt_particle_kept_hit_fraction": (
            lowpt_particle_df["kept_hit_fraction"].to_numpy(dtype=np.float64) if "kept_hit_fraction" in lowpt_particle_df else np.array([], dtype=np.float64)
        ),
        "lowpt_hit_r": lowpt_hit_df["r"].to_numpy(dtype=np.float64) if "r" in lowpt_hit_df else np.array([], dtype=np.float64),
        "lowpt_hit_eta": lowpt_hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in lowpt_hit_df else np.array([], dtype=np.float64),
        "lowpt_hit_phi": lowpt_hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in lowpt_hit_df else np.array([], dtype=np.float64),
        "lowpt_hit_score": lowpt_hit_df["score"].to_numpy(dtype=np.float64) if "score" in lowpt_hit_df else np.array([], dtype=np.float64),
        "lowpt_hit_min_distance_to_valid": (
            lowpt_hit_df["min_distance_to_valid_same_layer"].to_numpy(dtype=np.float64)
            if "min_distance_to_valid_same_layer" in lowpt_hit_df
            else np.array([], dtype=np.float64)
        ),
        "noise_hit_r": noise_hit_df["r"].to_numpy(dtype=np.float64) if "r" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_eta": noise_hit_df["eta"].to_numpy(dtype=np.float64) if "eta" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_phi": noise_hit_df["phi"].to_numpy(dtype=np.float64) if "phi" in noise_hit_df else np.array([], dtype=np.float64),
        "noise_hit_score": noise_hit_df["score"].to_numpy(dtype=np.float64) if "score" in noise_hit_df else np.array([], dtype=np.float64),
    }


def _write_prefilter_lowpt_noise_summary_csv(out_path: Path, samples: BandPrefilterLowPtNoiseSamples) -> None:
    feature_labels = {
        "event_prefilter_lowpt_hits": "Pre-filter hits from sub-threshold pT particles / event",
        "event_prefilter_noise_hits": "Pre-filter pure-noise hits / event",
        "event_fraction_prefilter_lowpt": "Fraction of source hits from sub-threshold pT particles",
        "event_total_raw_hits": "Total raw hits / event",
        "event_total_valid_raw_hits": "Valid raw hits / event",
        "event_prefilter_lowpt_source_particles": "Low-pT source particles / event",
        "event_fraction_all_hits_lowpt": "Fraction of all raw hits from low-pT particles",
        "event_fraction_all_hits_noise": "Fraction of all raw hits from pure noise",
        "event_prefilter_lowpt_near_valid_hits": "Low-pT hits near valid hits / event",
        "event_prefilter_lowpt_isolated_hits": "Isolated low-pT hits / event",
        "event_lowpt_hit_near_valid_fraction": "Fraction of low-pT hits near valid hits",
        "event_lowpt_hit_kept_fraction": "Fraction of low-pT hits kept",
        "event_lowpt_hit_kept_fraction_near_valid": "Kept fraction for near-valid low-pT hits",
        "event_lowpt_hit_kept_fraction_isolated": "Kept fraction for isolated low-pT hits",
        "event_valid_hit_recall": "Valid raw-hit recall at hit-filter threshold",
        "event_filtered_contamination": "Filtered-hit contamination at hit-filter threshold",
        "lowpt_particle_pt": "Source-particle pT [GeV]",
        "lowpt_particle_eta": "Source-particle eta",
        "lowpt_particle_num_hits_pre": "Source-particle hits in pre-filter event",
        "lowpt_particle_num_lowpt_hits_total": "Pre-filter hits per low-pT source particle",
        "lowpt_particle_kept_hit_fraction": "Fraction of source-particle hits kept by filter",
        "lowpt_hit_r": "Pre-filter low-pT-source hit r [m]",
        "lowpt_hit_eta": "Pre-filter low-pT-source hit eta",
        "lowpt_hit_phi": "Pre-filter low-pT-source hit phi [rad]",
        "lowpt_hit_score": "Pre-filter low-pT-source hit score",
        "lowpt_hit_min_distance_to_valid": "Low-pT-hit distance to valid hit in same layer [m]",
        "noise_hit_r": "Pre-filter pure-noise hit r [m]",
        "noise_hit_eta": "Pre-filter pure-noise hit eta",
        "noise_hit_phi": "Pre-filter pure-noise hit phi [rad]",
        "noise_hit_score": "Pre-filter pure-noise hit score",
    }
    rows: list[list[object]] = []
    for band_name in BAND_NAMES:
        feature_values = _prefilter_lowpt_noise_feature_map(samples, band_name)
        totals = samples.totals[band_name]
        for feature_key, values in feature_values.items():
            stats = _feature_stats(values)
            rows.append(
                [
                    samples.split,
                    band_name,
                    feature_key,
                    feature_labels[feature_key],
                    int(values.size),
                    totals["num_events"],
                    totals["num_prefilter_lowpt_hits"],
                    totals["num_prefilter_noise_hits"],
                    totals["num_lowpt_source_particles"],
                    stats["min"],
                    stats["p05"],
                    stats["median"],
                    stats["mean"],
                    stats["std"],
                    stats["p95"],
                    stats["max"],
                ]
            )
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "band",
                "feature",
                "label",
                "sample_size",
                "num_events_in_band",
                "num_prefilter_lowpt_hits_in_band",
                "num_prefilter_noise_hits_in_band",
                "num_lowpt_source_particles_in_band",
                "min",
                "p05",
                "median",
                "mean",
                "std",
                "p95",
                "max",
            ]
        )
        writer.writerows(rows)


def _plot_prefilter_lowpt_noise_comparison(samples: BandPrefilterLowPtNoiseSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_total_raw_hits": "Raw hits / event",
        "event_prefilter_lowpt_source_particles": "Low-pT source particles / event",
        "event_fraction_all_hits_lowpt": "Fraction of all hits from low-pT particles",
        "event_valid_hit_recall": "Valid raw-hit recall",
        "event_filtered_contamination": "Filtered-hit contamination",
        "event_prefilter_lowpt_hits": "Low-pT-source hits / event",
        "event_prefilter_noise_hits": "Pure-noise hits / event",
        "event_lowpt_hit_near_valid_fraction": "Fraction of low-pT hits near valid hits",
        "lowpt_particle_pt": "Low-pT source-particle pT [GeV]",
        "lowpt_particle_eta": "Low-pT source-particle eta",
        "lowpt_particle_num_hits_pre": "Low-pT source-particle hits / event",
        "lowpt_particle_num_lowpt_hits_total": "Hits per low-pT source particle",
        "lowpt_particle_kept_hit_fraction": "Kept-hit fraction per low-pT source particle",
        "lowpt_hit_r": "Low-pT-source hit r [m]",
        "lowpt_hit_eta": "Low-pT-source hit eta",
        "lowpt_hit_phi": "Low-pT-source hit phi [rad]",
        "lowpt_hit_score": "Low-pT-source hit score",
        "lowpt_hit_min_distance_to_valid": "Low-pT-hit distance to valid hit [m]",
        "noise_hit_r": "Pure-noise hit r [m]",
        "noise_hit_eta": "Pure-noise hit eta",
        "noise_hit_phi": "Pure-noise hit phi [rad]",
        "noise_hit_score": "Pure-noise hit score",
    }
    feature_order = [
        "event_total_raw_hits",
        "event_prefilter_lowpt_source_particles",
        "event_fraction_all_hits_lowpt",
        "event_valid_hit_recall",
        "event_filtered_contamination",
        "event_prefilter_lowpt_hits",
        "event_prefilter_noise_hits",
        "event_lowpt_hit_near_valid_fraction",
        "lowpt_particle_pt",
        "lowpt_particle_eta",
        "lowpt_particle_num_hits_pre",
        "lowpt_particle_num_lowpt_hits_total",
        "lowpt_particle_kept_hit_fraction",
        "lowpt_hit_r",
        "lowpt_hit_eta",
        "lowpt_hit_phi",
        "lowpt_hit_score",
        "lowpt_hit_min_distance_to_valid",
        "noise_hit_r",
        "noise_hit_score",
    ]
    per_band_features = {band_name: _prefilter_lowpt_noise_feature_map(samples, band_name) for band_name in BAND_NAMES}

    fig, axes = plt.subplots(5, 4, figsize=(15.8, 16.4), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    axes_flat = axes.ravel()
    legend_handles = []
    legend_labels = []
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key.endswith("_pt"):
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif feature_key.endswith("_phi"):
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        elif "fraction" in feature_key:
            bins, xscale = np.linspace(0.0, 1.0, 51), "linear"
        elif "num_hits" in feature_key or feature_key.endswith("_hits"):
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"
        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=num_events_by_band["lower_band"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=num_events_by_band["upper_band"])
        elif feature_key.startswith("lowpt_particle_"):
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.lowpt_particle_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_lowpt_source_particles"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.lowpt_particle_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_lowpt_source_particles"],
                num_events=num_events_by_band["upper_band"],
            )
        elif feature_key.startswith("lowpt_hit_"):
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.lowpt_hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_prefilter_lowpt_hits"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.lowpt_hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_prefilter_lowpt_hits"],
                num_events=num_events_by_band["upper_band"],
            )
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.noise_hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_prefilter_noise_hits"],
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.noise_hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_prefilter_noise_hits"],
                num_events=num_events_by_band["upper_band"],
            )
        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label="lower band",
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label="upper band",
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = ["lower band", "upper band"]
        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_band_mixed_hist_ylabel(feature_key))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=2, frameon=False)
    fig.suptitle(f"{samples.split}: pre-filter low-pT-source hits versus pure noise by post-filter band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_lowpt_noise_filter_status(samples: BandPrefilterLowPtNoiseSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(3, 4, figsize=(15.8, 10.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.40, wspace=0.30)
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    panel_specs = [
        ("lowpt_hit", "r", "Low-pT-source hit r [m]"),
        ("lowpt_hit", "eta", "Low-pT-source hit eta"),
        ("lowpt_hit", "phi", "Low-pT-source hit phi [rad]"),
        ("lowpt_hit", "score", "Low-pT-source hit score"),
        ("lowpt_parent", "parent_pt", "Parent low-pT particle pT [GeV]"),
        ("lowpt_parent", "parent_eta", "Parent low-pT particle eta"),
        ("lowpt_parent", "parent_num_hits_pre", "Parent low-pT particle hits / event"),
        ("lowpt_parent", "parent_num_lowpt_hits_total", "Hits per parent low-pT particle"),
        ("noise_hit", "r", "Pure-noise hit r [m]"),
        ("noise_hit", "eta", "Pure-noise hit eta"),
        ("noise_hit", "phi", "Pure-noise hit phi [rad]"),
        ("noise_hit", "score", "Pure-noise hit score"),
    ]
    status_styles = {"kept": "-", "removed": "--"}
    legend_handles = []
    legend_labels = []

    def _status_source_df(source_name: str, band_name: str) -> pd.DataFrame:
        if source_name in {"lowpt_hit", "lowpt_parent"}:
            return samples.lowpt_hit_samples[band_name]
        if source_name == "noise_hit":
            return samples.noise_hit_samples[band_name]
        raise ValueError(f"Unknown status-plot source_name={source_name!r}")

    def _status_total(source_name: str, band_name: str, status: str) -> int:
        metrics = samples.event_metrics[band_name]
        if source_name in {"lowpt_hit", "lowpt_parent"}:
            key = "event_prefilter_lowpt_hits_kept" if status == "kept" else "event_prefilter_lowpt_hits_removed"
            return int(np.sum(metrics[key]))
        if source_name == "noise_hit":
            key = "event_prefilter_noise_hits_kept" if status == "kept" else "event_prefilter_noise_hits_removed"
            return int(np.sum(metrics[key]))
        raise ValueError(f"Unknown status-plot source_name={source_name!r}")

    for ax, (sample_key, field, title) in zip(axes.ravel(), panel_specs):
        combined = np.array([], dtype=np.float64)
        for band_name in BAND_NAMES:
            sample_df = _status_source_df(sample_key, band_name)
            for status, linestyle in status_styles.items():
                values = sample_df.loc[sample_df["filter_status"].astype(str) == status, field].to_numpy(dtype=np.float64)
                combined = _combined_values(combined, values)

        if field.endswith("_pt"):
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif field.endswith("_phi") or field == "phi":
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        elif "fraction" in field:
            bins, xscale = np.linspace(0.0, 1.0, 51), "linear"
        elif "num_hits" in field:
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"

        ax.cla()
        for band_name in BAND_NAMES:
            sample_df = _status_source_df(sample_key, band_name)
            for status, linestyle in status_styles.items():
                values = sample_df.loc[sample_df["filter_status"].astype(str) == status, field].to_numpy(dtype=np.float64)
                hist = ax.hist(
                    values,
                    bins=bins,
                    weights=_sampled_population_hist_weights(
                        len(values),
                        sampled_population_size=len(values),
                        total_population=_status_total(sample_key, band_name, status),
                        num_events=num_events_by_band[band_name],
                    ),
                    density=_hist_density(),
                    histtype="step",
                    linewidth=1.4,
                    color=BAND_COLORS[band_name],
                    linestyle=linestyle,
                    label=f"{band_name.replace('_', ' ')} | {status}",
                )
                if len(legend_handles) < 4:
                    legend_handles.append(hist[2][0])
                    legend_labels.append(f"{band_name.replace('_', ' ')} | {status}")
        ax.set_title(title, fontsize=10.8)
        ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=4, frameon=False)
    fig.suptitle(f"{samples.split}: pre-filter low-pT/noise hit distributions split by hit-filter outcome", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_lowpt_event_diagnostics(samples: BandPrefilterLowPtNoiseSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    diagnostics = samples.event_diagnostics.copy()
    if diagnostics.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.28)
    axes = axes.ravel()

    hist_specs = [
        ("event_fraction_all_hits_lowpt", "Fraction of all raw hits from low-pT particles"),
        ("event_prefilter_lowpt_source_particles", "Low-pT source particles / event"),
        ("event_lowpt_hit_near_valid_fraction", "Fraction of low-pT hits near valid hits"),
    ]
    for ax, (column, title) in zip(axes[:3], hist_specs, strict=True):
        lower_vals = diagnostics.loc[diagnostics["band"] == "lower_band", column].to_numpy(dtype=np.float64)
        upper_vals = diagnostics.loc[diagnostics["band"] == "upper_band", column].to_numpy(dtype=np.float64)
        combined = _combined_values(lower_vals, upper_vals)
        if "fraction" in column:
            bins = np.linspace(0.0, 1.0, 41)
        else:
            bins = _count_bins(combined)
        ax.hist(
            lower_vals,
            bins=bins,
            weights=_hist_weights(
                len(lower_vals),
                num_events=samples.totals["lower_band"]["num_events"],
                allow_count_per_event=True,
            ),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label="lower band",
        )
        ax.hist(
            upper_vals,
            bins=bins,
            weights=_hist_weights(
                len(upper_vals),
                num_events=samples.totals["upper_band"]["num_events"],
                allow_count_per_event=True,
            ),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label="upper band",
        )
        ax.set_title(title)
        ax.set_ylabel(_event_hist_ylabel(per_band=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    scatter_specs = [
        ("event_fraction_all_hits_lowpt", "Fraction of all raw hits from low-pT particles", "event_valid_hit_recall", "Low-pT raw-hit fraction vs valid-hit recall", "Valid raw-hit recall"),
        ("event_fraction_all_hits_lowpt", "Fraction of all raw hits from low-pT particles", "event_filtered_contamination", "Low-pT raw-hit fraction vs filtered contamination", "Filtered-hit contamination"),
        ("event_lowpt_hit_near_valid_fraction", "Fraction of low-pT hits near valid hits", "event_lowpt_hit_kept_fraction", "Near-valid low-pT fraction vs low-pT kept fraction", "Low-pT hit kept fraction"),
    ]
    for ax, (x_col, xlabel, y_col, title, ylabel) in zip(axes[3:], scatter_specs, strict=True):
        for band_name in BAND_NAMES:
            mask = diagnostics["band"] == band_name
            ax.scatter(
                diagnostics.loc[mask, x_col],
                diagnostics.loc[mask, y_col],
                s=12.0,
                alpha=0.58,
                color=BAND_COLORS[band_name],
                edgecolors="none",
                rasterized=True,
                label=band_name.replace("_", " "),
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"{samples.split}: event-level low-pT diagnostics by post-filter band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_lowpt_proximity_comparison(samples: BandPrefilterLowPtNoiseSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.8), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.30)
    lowpt_by_band = {band_name: samples.lowpt_hit_samples[band_name] for band_name in BAND_NAMES}
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}
    panel_specs = [
        ("score", "Low-pT-hit score"),
        ("min_distance_to_valid_same_layer", "Low-pT-hit distance to valid hit [m]"),
        ("r", "Low-pT-hit r [m]"),
        ("eta", "Low-pT-hit eta"),
        ("parent_pt", "Parent low-pT particle pT [GeV]"),
        ("parent_eta", "Parent low-pT particle eta"),
        ("parent_num_hits_pre", "Parent low-pT particle hits / event"),
        ("parent_num_lowpt_hits_total", "Hits per low-pT parent"),
    ]
    bucket_styles = {"near_valid": "-", "isolated": "--"}
    legend_handles = []
    legend_labels = []
    bucket_total_keys = {
        "near_valid": "event_prefilter_lowpt_near_valid_hits",
        "isolated": "event_prefilter_lowpt_isolated_hits",
    }

    for ax, (field, title) in zip(axes.ravel(), panel_specs, strict=True):
        combined = np.array([], dtype=np.float64)
        for band_name in BAND_NAMES:
            sample_df = lowpt_by_band[band_name]
            for bucket in bucket_styles:
                values = sample_df.loc[sample_df["proximity_bucket"].astype(str) == bucket, field].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                combined = _combined_values(combined, values)

        if field.endswith("_pt"):
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif field.endswith("_eta") or field == "eta":
            bins, xscale = _continuous_bins(combined), "linear"
        elif "num_hits" in field:
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"

        for band_name in BAND_NAMES:
            sample_df = lowpt_by_band[band_name]
            for bucket, linestyle in bucket_styles.items():
                values = sample_df.loc[sample_df["proximity_bucket"].astype(str) == bucket, field].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                hist = ax.hist(
                    values,
                    bins=bins,
                    weights=_sampled_population_hist_weights(
                        len(values),
                        sampled_population_size=len(values),
                        total_population=int(np.sum(samples.event_metrics[band_name][bucket_total_keys[bucket]])),
                        num_events=num_events_by_band[band_name],
                    ),
                    density=_hist_density(),
                    histtype="step",
                    linewidth=1.4,
                    color=BAND_COLORS[band_name],
                    linestyle=linestyle,
                    label=f"{band_name.replace('_', ' ')} | {bucket.replace('_', ' ')}",
                )
                if len(legend_handles) < 4:
                    legend_handles.append(hist[2][0])
                    legend_labels.append(f"{band_name.replace('_', ' ')} | {bucket.replace('_', ' ')}")
        ax.set_title(title, fontsize=10.8)
        ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.952), ncol=4, frameon=False)
    fig.suptitle(f"{samples.split}: low-pT-hit proximity to valid hits by post-filter band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_lowpt_region_diagnostics(
    samples: BandPrefilterLowPtNoiseSamples,
    output_base: Path,
    top_region_keys: int,
) -> None:
    plt = _get_pyplot()
    region_df = samples.region_summary.copy()
    if region_df.empty:
        return

    pivot_ratio = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="lowpt_to_valid_ratio",
        aggfunc="first",
    )
    pivot_kept = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="lowpt_kept_fraction",
        aggfunc="first",
    )
    pivot_fraction = region_df.pivot_table(
        index=["volume_id", "layer_id"],
        columns="band",
        values="lowpt_hit_fraction",
        aggfunc="first",
    )
    for pivot in (pivot_ratio, pivot_kept, pivot_fraction):
        for band_name in BAND_NAMES:
            if band_name not in pivot.columns:
                pivot[band_name] = np.nan

    ranking = (pivot_ratio[BAND_NAMES[0]].fillna(0.0) - pivot_ratio[BAND_NAMES[1]].fillna(0.0)).abs().sort_values(ascending=False)
    selected_keys = list(ranking.head(max(1, top_region_keys)).index)
    labels = [f"V{int(volume_id)}/L{int(layer_id)}" for volume_id, layer_id in selected_keys]
    y = np.arange(len(selected_keys), dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, max(5.6, 0.42 * len(selected_keys) + 2.4)), constrained_layout=False)
    fig.subplots_adjust(top=0.88, wspace=0.38)
    panel_specs = [
        (pivot_ratio, "Low-pT / valid-hit ratio"),
        (pivot_fraction, "Low-pT hit fraction"),
        (pivot_kept, "Low-pT kept fraction"),
    ]
    for ax, (pivot, xlabel) in zip(axes, panel_specs, strict=True):
        lower_vals = np.array([pivot.loc[key, BAND_NAMES[0]] for key in selected_keys], dtype=np.float64)
        upper_vals = np.array([pivot.loc[key, BAND_NAMES[1]] for key in selected_keys], dtype=np.float64)
        ax.barh(y + 0.18, lower_vals, height=0.34, color=BAND_COLORS[BAND_NAMES[0]], label=BAND_NAMES[0].replace("_", " "), alpha=0.85)
        ax.barh(y - 0.18, upper_vals, height=0.34, color=BAND_COLORS[BAND_NAMES[1]], label=BAND_NAMES[1].replace("_", " "), alpha=0.85)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45, axis="x")
        ax.set_axisbelow(True)
    axes[0].set_title("Top regions by low-pT occupancy difference")
    axes[1].set_title("Low-pT fraction by region")
    axes[2].set_title("Low-pT kept fraction by region")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{samples.split}: region-level low-pT occupancy diagnostics", fontsize=13, y=0.965)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _collect_band_prefilter_truth_nonreco_samples(
    cfg: dict,
    assignments: BandAssignments,
    *,
    max_particle_samples_per_band: int,
    max_hit_samples_per_band: int,
    rng_seed: int,
) -> BandPrefilterTruthNonrecoSamples:
    data_cfg = cfg["data"]
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    min_pt = float(data_cfg["particle_min_pt"])
    max_abs_eta = float(data_cfg["particle_max_abs_eta"])
    min_num_hits = int(data_cfg["particle_min_num_hits"])
    rng = np.random.default_rng(rng_seed + 5000)

    particle_samples = {
        band_name: pd.DataFrame(
            columns=["pt", "eta", "phi", "num_hits_pre", "num_unique_layers", "primary_fail_reason", "fail_reason", "_sample_key"]
        )
        for band_name in BAND_NAMES
    }
    event_metrics = {
        band_name: {
            "event_total_nonreco": [],
            "event_total_particles": [],
            "event_num_fail_pt": [],
            "event_num_fail_contains_pt": [],
            "event_num_fail_eta": [],
            "event_num_fail_min_hits": [],
            "event_num_fail_multiple": [],
            "event_fraction_nonreco": [],
        }
        for band_name in BAND_NAMES
    }
    totals = {
        band_name: {
            "num_events": 0,
            "num_nonreco_particles": 0,
            "num_fail_pt": 0,
            "num_fail_contains_pt": 0,
            "num_fail_eta": 0,
            "num_fail_min_hits": 0,
            "num_fail_multiple": 0,
        }
        for band_name in BAND_NAMES
    }
    noise_hit_samples = {
        band_name: pd.DataFrame(columns=["r", "eta", "phi", "z", "_sample_key"])
        for band_name in BAND_NAMES
    }
    noise_event_metrics = {
        band_name: {"event_total_noise_hits": [], "event_total_hits": [], "event_noise_fraction": []}
        for band_name in BAND_NAMES
    }

    print(f"[{assignments.split}] collecting pre-filter truth nonreco particle and noise hit samples from raw events")
    for event_idx, band_code in enumerate(assignments.band_codes):
        band_name = BAND_NAMES[int(band_code)]
        hits_raw, particles_raw = _load_raw_trackml_event(
            assignments.split_dir, assignments.event_names[event_idx], hit_volume_ids
        )

        particles_all = particles_raw.copy()
        hit_counts_pre = hits_raw["particle_id"].value_counts()
        particles_all["num_hits_pre"] = particles_all["particle_id"].map(hit_counts_pre).fillna(0).astype(np.int32)

        unique_layer_counts = (
            hits_raw.assign(_lk=hits_raw["volume_id"] * 1000 + hits_raw["layer_id"])
            .groupby("particle_id")["_lk"]
            .nunique()
        )
        particles_all["num_unique_layers"] = particles_all["particle_id"].map(unique_layer_counts).fillna(0).astype(np.int32)

        fail_pt = particles_all["pt"].to_numpy(dtype=np.float64, copy=False) <= min_pt
        fail_eta = np.abs(particles_all["eta"].to_numpy(dtype=np.float64, copy=False)) >= max_abs_eta
        fail_min_hits = particles_all["num_hits_pre"].to_numpy(dtype=np.int32, copy=False) < min_num_hits
        fail_reason, primary_fail_reason = _compose_fail_reason_labels(fail_pt, fail_eta, fail_min_hits)
        particles_all["fail_reason"] = fail_reason
        particles_all["primary_fail_reason"] = primary_fail_reason

        is_nonreco = fail_pt | fail_eta | fail_min_hits
        nonreco_particles = particles_all.loc[
            is_nonreco, ["pt", "eta", "phi", "num_hits_pre", "num_unique_layers", "primary_fail_reason", "fail_reason"]
        ].copy()

        prim = nonreco_particles["primary_fail_reason"].to_numpy(dtype=str, copy=False)
        num_total = len(particles_all)
        num_nonreco = int(is_nonreco.sum())
        num_fail_pt = int(np.sum(prim == "pt"))
        num_fail_contains_pt = int(np.sum(fail_pt))
        num_fail_eta = int(np.sum(prim == "eta"))
        num_fail_min_hits = int(np.sum(prim == "min_hits"))
        num_fail_multiple = int(np.sum(prim == "multiple"))

        event_metrics[band_name]["event_total_nonreco"].append(num_nonreco)
        event_metrics[band_name]["event_total_particles"].append(num_total)
        event_metrics[band_name]["event_num_fail_pt"].append(num_fail_pt)
        event_metrics[band_name]["event_num_fail_contains_pt"].append(num_fail_contains_pt)
        event_metrics[band_name]["event_num_fail_eta"].append(num_fail_eta)
        event_metrics[band_name]["event_num_fail_min_hits"].append(num_fail_min_hits)
        event_metrics[band_name]["event_num_fail_multiple"].append(num_fail_multiple)
        event_metrics[band_name]["event_fraction_nonreco"].append(
            float(num_nonreco) / float(num_total) if num_total > 0 else 0.0
        )
        totals[band_name]["num_events"] += 1
        totals[band_name]["num_nonreco_particles"] += num_nonreco
        totals[band_name]["num_fail_pt"] += num_fail_pt
        totals[band_name]["num_fail_contains_pt"] += num_fail_contains_pt
        totals[band_name]["num_fail_eta"] += num_fail_eta
        totals[band_name]["num_fail_min_hits"] += num_fail_min_hits
        totals[band_name]["num_fail_multiple"] += num_fail_multiple

        particle_samples[band_name] = _priority_sample(
            particle_samples[band_name], nonreco_particles, max_particle_samples_per_band, rng
        )

        # Noise hits: particle_id == 0
        noise_hits = hits_raw.loc[hits_raw["particle_id"] == 0, ["r", "eta", "phi", "z"]].copy()
        num_noise = len(noise_hits)
        num_all_hits = len(hits_raw)
        noise_event_metrics[band_name]["event_total_noise_hits"].append(num_noise)
        noise_event_metrics[band_name]["event_total_hits"].append(num_all_hits)
        noise_event_metrics[band_name]["event_noise_fraction"].append(
            float(num_noise) / float(num_all_hits) if num_all_hits > 0 else 0.0
        )
        noise_hit_samples[band_name] = _priority_sample(
            noise_hit_samples[band_name], noise_hits, max_hit_samples_per_band, rng
        )

        if (event_idx + 1) % 50 == 0 or (event_idx + 1) == len(assignments.band_codes):
            print(f"[{assignments.split}] pre-filter truth nonreco: processed {event_idx + 1}/{len(assignments.band_codes)} events")

    for band_name in BAND_NAMES:
        for key in list(event_metrics[band_name].keys()):
            event_metrics[band_name][key] = np.array(event_metrics[band_name][key], dtype=np.float64)
        for key in list(noise_event_metrics[band_name].keys()):
            noise_event_metrics[band_name][key] = np.array(noise_event_metrics[band_name][key], dtype=np.float64)

    return BandPrefilterTruthNonrecoSamples(
        split=assignments.split,
        particle_samples=particle_samples,
        event_metrics=event_metrics,
        totals=totals,
        noise_hit_samples=noise_hit_samples,
        noise_event_metrics=noise_event_metrics,
    )


def _plot_prefilter_truth_nonreco_comparison(samples: BandPrefilterTruthNonrecoSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    reason_order = ["pt", "eta", "min_hits"]
    feature_specs = [
        ("pt", "Source particle pT [GeV]"),
        ("eta", "Source particle eta"),
        ("phi", "Source particle phi"),
        ("num_hits_pre", "Truth hits per particle"),
        ("num_unique_layers", "Unique detector layers hit"),
    ]
    fig, axes = plt.subplots(len(reason_order), len(feature_specs), figsize=(19.0, 10.2), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.30)
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    for row_idx, reason in enumerate(reason_order):
        for col_idx, (feature_key, title) in enumerate(feature_specs):
            ax = axes[row_idx, col_idx]
            lower_df = samples.particle_samples["lower_band"]
            upper_df = samples.particle_samples["upper_band"]
            lower_vals = lower_df.loc[lower_df["primary_fail_reason"].astype(str) == reason, feature_key].to_numpy(dtype=np.float64)
            upper_vals = upper_df.loc[upper_df["primary_fail_reason"].astype(str) == reason, feature_key].to_numpy(dtype=np.float64)
            lower_reason_count = int(samples.totals["lower_band"].get(f"num_fail_{reason}", 0))
            upper_reason_count = int(samples.totals["upper_band"].get(f"num_fail_{reason}", 0))
            combined = _combined_values(lower_vals, upper_vals)
            if feature_key == "pt":
                bins, xscale = _feature_bins_and_scale("particle_pt", combined)
            elif feature_key in {"num_hits_pre", "num_unique_layers"}:
                bins, xscale = _count_bins(combined), "linear"
            elif feature_key == "phi":
                bins, xscale = _feature_bins_and_scale("particle_phi", combined)
            else:
                bins, xscale = _continuous_bins(combined), "linear"
            ax.hist(
                lower_vals,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(lower_vals),
                    sampled_population_size=len(lower_vals),
                    total_population=lower_reason_count,
                    num_events=num_events_by_band["lower_band"],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5, color=BAND_COLORS["lower_band"], label="lower band",
            )
            ax.hist(
                upper_vals,
                bins=bins,
                weights=_sampled_population_hist_weights(
                    len(upper_vals),
                    sampled_population_size=len(upper_vals),
                    total_population=upper_reason_count,
                    num_events=num_events_by_band["upper_band"],
                ),
                density=_hist_density(),
                histtype="step",
                linewidth=1.5, color=BAND_COLORS["upper_band"], label="upper band",
            )
            ax.set_title(f"{reason}: {title}", fontsize=10.5)
            ax.set_ylabel(_hist_ylabel(allow_count_per_event=True))
            ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
            ax.set_axisbelow(True)
            if xscale == "log":
                ax.set_xscale("log")
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{samples.split}: pre-filter truth non-reconstructable particle distributions by fail reason",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_truth_nonreco_event_metrics(samples: BandPrefilterTruthNonrecoSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    metric_specs = [
        ("event_total_nonreco", "Total nonreco particles / event"),
        ("event_num_fail_pt", "Fail-pT only / event"),
        ("event_num_fail_eta", "Fail-|η| only / event"),
        ("event_num_fail_min_hits", "Fail-min-hits only / event"),
        ("event_fraction_nonreco", "Nonreco fraction of all particles"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(19.0, 4.2), constrained_layout=True)
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    for ax, (metric_key, title) in zip(axes, metric_specs):
        lower_vals = samples.event_metrics["lower_band"].get(metric_key, np.array([], dtype=np.float64))
        upper_vals = samples.event_metrics["upper_band"].get(metric_key, np.array([], dtype=np.float64))
        combined = _combined_values(lower_vals, upper_vals)
        if metric_key == "event_fraction_nonreco":
            hi = float(np.percentile(combined, 99.5)) * 1.05 if combined.size > 0 else 1.0
            bins = np.linspace(0.0, min(1.0, hi), 51)
        else:
            bins = _count_bins(combined)
        ax.hist(
            lower_vals,
            bins=bins,
            weights=_band_hist_weights(len(lower_vals), num_events=num_events_by_band["lower_band"]),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["lower_band"], label="lower band",
        )
        ax.hist(
            upper_vals,
            bins=bins,
            weights=_band_hist_weights(len(upper_vals), num_events=num_events_by_band["upper_band"]),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["upper_band"], label="upper band",
        )
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(_event_hist_ylabel())
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{samples.split}: pre-filter truth nonreco event-level counts by band",
        fontsize=13,
    )
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_truth_noise_comparison(samples: BandPrefilterTruthNonrecoSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    # 2×3 grid: two event-level panels then four spatial hit-feature panels
    feature_specs = [
        ("event_total_noise_hits", "Total noise hits / event", "event"),
        ("event_noise_fraction", "Noise fraction of all hits / event", "event"),
        ("r", "Noise hit r [m]", "hit"),
        ("eta", "Noise hit eta", "hit"),
        ("phi", "Noise hit phi [rad]", "hit"),
        ("z", "Noise hit z [m]", "hit"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), constrained_layout=False)
    fig.subplots_adjust(top=0.90, hspace=0.38, wspace=0.30)
    axes_flat = axes.ravel()
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}

    for ax, (feature_key, title, source) in zip(axes_flat, feature_specs):
        if source == "event":
            lower_vals = samples.noise_event_metrics["lower_band"].get(feature_key, np.array([], dtype=np.float64))
            upper_vals = samples.noise_event_metrics["upper_band"].get(feature_key, np.array([], dtype=np.float64))
        else:
            lower_df = samples.noise_hit_samples["lower_band"]
            upper_df = samples.noise_hit_samples["upper_band"]
            lower_vals = lower_df[feature_key].to_numpy(dtype=np.float64) if feature_key in lower_df else np.array([], dtype=np.float64)
            upper_vals = upper_df[feature_key].to_numpy(dtype=np.float64) if feature_key in upper_df else np.array([], dtype=np.float64)
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key == "event_noise_fraction":
            hi = float(np.percentile(combined, 99.5)) * 1.05 if combined.size > 0 else 1.0
            bins = np.linspace(0.0, min(1.0, hi), 51)
        elif feature_key == "phi":
            bins = np.linspace(-np.pi, np.pi, 61)
        elif source == "event":
            bins = _count_bins(combined)
        else:
            bins = _continuous_bins(combined)
        if source == "event":
            lower_weights = _band_hist_weights(len(lower_vals), num_events=num_events_by_band["lower_band"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=num_events_by_band["upper_band"])
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.noise_hit_samples["lower_band"]),
                total_population=int(np.sum(samples.noise_event_metrics["lower_band"]["event_total_noise_hits"])),
                num_events=num_events_by_band["lower_band"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.noise_hit_samples["upper_band"]),
                total_population=int(np.sum(samples.noise_event_metrics["upper_band"]["event_total_noise_hits"])),
                num_events=num_events_by_band["upper_band"],
            )
        ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["lower_band"], label="lower band",
        )
        ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["upper_band"], label="upper band",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(_event_hist_ylabel(per_band=True) if source == "event" else _hist_ylabel(allow_count_per_event=True))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
    axes_flat[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{samples.split}: pre-filter pure noise hit distributions by band", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefilter_lowpt_particle_kinematics(samples: BandPrefilterTruthNonrecoSamples, output_base: Path) -> None:
    """2×2 plot of pre-filter sub-threshold-pT particle kinematics by band.

    Population: all particles where the pT cut is among the fail reasons (i.e. fail_reason
    contains 'pt'), regardless of whether other cuts also apply.  Panels: pT, eta, phi,
    pre-filter hit count per particle.
    """
    plt = _get_pyplot()

    feature_specs = [
        ("pt",           "pT [GeV]"),
        ("eta",          "eta"),
        ("phi",          "phi [rad]"),
        ("num_hits_pre", "Truth hits per track"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)
    axes_flat = axes.ravel()
    num_events_by_band = {band_name: int(samples.totals[band_name]["num_events"]) for band_name in BAND_NAMES}
    total_pt_fail_by_band = {
        band_name: int(samples.totals[band_name].get("num_fail_contains_pt", samples.totals[band_name].get("num_fail_pt", 0)))
        for band_name in BAND_NAMES
    }

    for ax, (feature_key, title) in zip(axes_flat, feature_specs):
        lower_df = samples.particle_samples["lower_band"]
        upper_df = samples.particle_samples["upper_band"]

        # Keep all particles where pT is one of the fail reasons (includes "pt+eta" etc.)
        lower_vals = lower_df.loc[
            lower_df["fail_reason"].astype(str).str.contains("pt", regex=False), feature_key
        ].to_numpy(dtype=np.float64)
        upper_vals = upper_df.loc[
            upper_df["fail_reason"].astype(str).str.contains("pt", regex=False), feature_key
        ].to_numpy(dtype=np.float64)

        combined = _combined_values(lower_vals, upper_vals)
        if feature_key == "pt":
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif feature_key == "phi":
            bins, xscale = _feature_bins_and_scale("particle_phi", combined)
        elif feature_key == "num_hits_pre":
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"

        ax.hist(
            lower_vals,
            bins=bins,
            weights=_sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(lower_vals),
                total_population=total_pt_fail_by_band["lower_band"],
                num_events=num_events_by_band["lower_band"],
            ),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["lower_band"], label="lower band",
        )
        ax.hist(
            upper_vals,
            bins=bins,
            weights=_sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(upper_vals),
                total_population=total_pt_fail_by_band["upper_band"],
                num_events=num_events_by_band["upper_band"],
            ),
            density=_hist_density(),
            histtype="step",
            linewidth=1.5, color=BAND_COLORS["upper_band"], label="upper band",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(_band_mixed_hist_ylabel(feature_key))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    axes_flat[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{samples.split}: pre-filter sub-threshold-pT particles by band",
        fontsize=13,
    )
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_band_assignments_csv(out_path: Path, assignments: BandAssignments) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "event_index",
                "sample_id",
                "event_name",
                "band",
                "hits_after_filter",
                "num_reconstructable_tracks",
                "fitted_reconstructable_tracks",
                "residual_tracks",
                "band_position",
                "upper_band_posterior",
                "event_hits_per_track",
            ]
        )
        for event_index, sample_id, event_name, hits, particles, fitted, residual, band_code, band_position, upper_band_posterior in zip(
            assignments.event_indices,
            assignments.sample_ids,
            assignments.event_names,
            assignments.hits,
            assignments.particles,
            assignments.fitted_particles,
            assignments.residuals,
            assignments.band_codes,
            assignments.band_position,
            assignments.upper_band_posterior,
            strict=True,
        ):
            writer.writerow(
                [
                    assignments.split,
                    int(event_index),
                    int(sample_id),
                    event_name,
                    BAND_NAMES[int(band_code)],
                    int(hits),
                    int(particles),
                    float(fitted),
                    float(residual),
                    float(band_position),
                    float(upper_band_posterior),
                    float(hits) / max(float(particles), 1.0),
                ]
            )


def _write_band_feature_summary_csv(out_path: Path, samples: BandFeatureSamples) -> None:
    feature_labels = {
        "event_hits_per_track": "Event hits after filtering / reconstructable track",
        "hit_x": "Hit x [m]",
        "hit_y": "Hit y [m]",
        "hit_z": "Hit z [m]",
        "hit_r": "Hit r [m]",
        "hit_eta": "Hit eta",
        "hit_phi": "Hit phi [rad]",
        "particle_pt": "Particle pT [GeV]",
        "particle_eta": "Particle eta",
        "particle_phi": "Particle phi [rad]",
        "particle_num_hits_post": "Track hits after filtering",
        "particle_num_hits_pre": "Track hits before filtering",
        "particle_hit_retention": "Track hit retention fraction",
    }

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "band",
                "feature",
                "label",
                "sample_size",
                "num_events_in_band",
                "num_filtered_hits_in_band",
                "num_selected_hits_in_band",
                "num_reconstructable_tracks_in_band",
                "min",
                "p05",
                "median",
                "mean",
                "std",
                "p95",
                "max",
            ]
        )

        for band_name in BAND_NAMES:
            feature_values = _feature_map(samples, band_name)
            for feature_key, values in feature_values.items():
                stats = _feature_stats(values)
                totals = samples.totals[band_name]
                writer.writerow(
                    [
                        samples.split,
                        band_name,
                        feature_key,
                        feature_labels[feature_key],
                        int(values.size),
                        totals["num_events"],
                        totals["num_filtered_hits"],
                        totals["num_selected_hits"],
                        totals["num_reconstructable_tracks"],
                        stats["min"],
                        stats["p05"],
                        stats["median"],
                        stats["mean"],
                        stats["std"],
                        stats["p95"],
                        stats["max"],
                    ]
                )


def _write_truth_band_summary_csv(out_path: Path, samples: TruthBandSamples) -> None:
    feature_labels = {
        "event_true_tracks": "Reconstructable truth tracks per event",
        "event_true_hits": "True hits on reconstructable tracks per event",
        "event_true_hits_per_true_track": "True hits per reconstructable truth track",
        "true_hit_x": "True-hit x [m]",
        "true_hit_y": "True-hit y [m]",
        "true_hit_z": "True-hit z [m]",
        "true_hit_r": "True-hit r [m]",
        "true_hit_eta": "True-hit eta",
        "true_hit_phi": "True-hit phi [rad]",
        "true_track_pt": "Reconstructable truth-track pT [GeV]",
        "true_track_eta": "Reconstructable truth-track eta",
        "true_track_phi": "Reconstructable truth-track phi [rad]",
        "true_track_num_hits": "True hits per reconstructable truth track",
    }

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "band",
                "feature",
                "label",
                "sample_size",
                "num_events_in_band",
                "num_true_hits_in_band",
                "num_true_tracks_in_band",
                "min",
                "p05",
                "median",
                "mean",
                "std",
                "p95",
                "max",
            ]
        )

        for band_name in BAND_NAMES:
            feature_values = _truth_feature_map(samples, band_name)
            for feature_key, values in feature_values.items():
                stats = _feature_stats(values)
                totals = samples.totals[band_name]
                writer.writerow(
                    [
                        samples.split,
                        band_name,
                        feature_key,
                        feature_labels[feature_key],
                        int(values.size),
                        totals["num_events"],
                        totals["num_true_hits"],
                        totals["num_true_tracks"],
                        stats["min"],
                        stats["p05"],
                        stats["median"],
                        stats["mean"],
                        stats["std"],
                        stats["p95"],
                        stats["max"],
                    ]
                )


def _write_track_loss_summary_csv(out_path: Path, samples: BandTrackLossSamples) -> None:
    feature_labels = {
        "event_truth_tracks": "Truth reconstructable tracks / event",
        "event_surviving_tracks": "Surviving reconstructable tracks / event",
        "event_lost_tracks": "Lost reconstructable tracks / event",
        "event_lost_track_fraction": "Fraction of reconstructable tracks lost",
        "event_mean_hits_lost_per_truth_track": "Mean hits lost per reconstructable track / event",
        "event_mean_hit_retention_per_truth_track": "Mean hit retention per reconstructable track / event",
        "track_pt": "Reconstructable track pT [GeV]",
        "track_eta": "Reconstructable track eta",
        "track_phi": "Reconstructable track phi [rad]",
        "track_num_hits_pre": "Reconstructable track hits before filtering",
        "track_num_hits_post": "Reconstructable track hits after filtering",
        "track_num_hits_lost": "Reconstructable track hits lost",
        "track_hit_retention": "Reconstructable track hit retention",
        "track_survived_filter": "Reconstructable track survives filtering",
    }

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "band",
                "feature",
                "label",
                "sample_size",
                "num_events_in_band",
                "num_truth_tracks_in_band",
                "num_surviving_tracks_in_band",
                "num_lost_tracks_in_band",
                "min",
                "p05",
                "median",
                "mean",
                "std",
                "p95",
                "max",
            ]
        )

        for band_name in BAND_NAMES:
            feature_values = _track_loss_feature_map(samples, band_name)
            for feature_key, values in feature_values.items():
                stats = _feature_stats(values)
                totals = samples.totals[band_name]
                writer.writerow(
                    [
                        samples.split,
                        band_name,
                        feature_key,
                        feature_labels[feature_key],
                        int(values.size),
                        totals["num_events"],
                        totals["num_truth_tracks"],
                        totals["num_surviving_tracks"],
                        totals["num_lost_tracks"],
                        stats["min"],
                        stats["p05"],
                        stats["median"],
                        stats["mean"],
                        stats["std"],
                        stats["p95"],
                        stats["max"],
                    ]
                )


def _plot_band_scatter(assignments: BandAssignments, output_base: Path) -> None:
    plt = _get_pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    scatter_ax, resid_ax = axes

    for band_code, band_name in enumerate(BAND_NAMES):
        mask = assignments.band_codes == band_code
        scatter_ax.scatter(
            assignments.hits[mask],
            assignments.particles[mask],
            s=7.0,
            marker=".",
            alpha=0.68,
            color=BAND_COLORS[band_name],
            edgecolors="none",
            rasterized=True,
            label=f"{band_name.replace('_', ' ')} (n={int(mask.sum())})",
        )
        resid_ax.hist(
            assignments.residuals[mask],
            bins=_continuous_bins(assignments.residuals, n_bins=50, low_pct=0.0, high_pct=100.0),
            weights=_hist_weights(int(mask.sum()), num_events=int(mask.sum()), allow_count_per_event=True),
            density=_hist_density(),
            histtype="step",
            linewidth=1.4,
            color=BAND_COLORS[band_name],
            label=f"{band_name.replace('_', ' ')}",
        )
        resid_ax.axvline(assignments.band_centers[band_code], color=BAND_COLORS[band_name], linestyle="--", linewidth=1.0, alpha=0.85)

    x_line = np.linspace(float(assignments.hits.min()), float(assignments.hits.max()) * 1.02, 256)
    for band_code, band_name in enumerate(BAND_NAMES):
        scatter_ax.plot(
            x_line,
            assignments.band_intercepts[band_code] + assignments.model_slope * x_line,
            color=BAND_COLORS[band_name],
            linestyle="--",
            linewidth=1.05,
            alpha=0.9,
        )

    midpoint_intercept = float(np.mean(assignments.band_intercepts))
    scatter_ax.plot(
        x_line,
        midpoint_intercept + assignments.model_slope * x_line,
        color="#666666",
        linestyle=":",
        linewidth=0.9,
        alpha=0.85,
    )

    scatter_ax.set_xlabel("Hits after hit filtering")
    scatter_ax.set_ylabel("Reconstructable tracks per event")
    scatter_ax.set_title(f"{assignments.split}: post-filter event bands")
    scatter_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    scatter_ax.set_axisbelow(True)
    scatter_ax.legend(loc="best", frameon=False)

    resid_ax.set_xlabel("Residual tracks relative to assigned band line")
    resid_ax.set_ylabel(_event_hist_ylabel(per_band=True))
    resid_ax.set_title(f"{assignments.split}: residual split")
    resid_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    resid_ax.set_axisbelow(True)

    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_band_feature_comparison(samples: BandFeatureSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_hits_per_track": "Event hits / reconstructable track",
        "hit_x": "Hit x [m]",
        "hit_y": "Hit y [m]",
        "hit_z": "Hit z [m]",
        "hit_r": "Hit r [m]",
        "hit_eta": "Hit eta",
        "hit_phi": "Hit phi [rad]",
        "particle_pt": "Particle pT [GeV]",
        "particle_eta": "Particle eta",
        "particle_phi": "Particle phi [rad]",
        "particle_num_hits_post": "Track hits after filtering",
        "particle_num_hits_pre": "Track hits before filtering",
        "particle_hit_retention": "Track hit retention fraction",
    }
    feature_order = [
        "event_hits_per_track",
        "hit_x",
        "hit_y",
        "hit_z",
        "hit_r",
        "hit_eta",
        "hit_phi",
        "particle_pt",
        "particle_eta",
        "particle_phi",
        "particle_num_hits_post",
        "particle_num_hits_pre",
        "particle_hit_retention",
    ]

    per_band_features = {band_name: _feature_map(samples, band_name) for band_name in BAND_NAMES}
    fig, axes = plt.subplots(4, 4, figsize=(15.4, 12.8), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    axes_flat = axes.ravel()

    legend_handles = []
    legend_labels = []
    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        combined = _combined_values(lower_vals, upper_vals)
        bins, xscale = _feature_bins_and_scale(feature_key, combined)

        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=samples.totals["lower_band"]["num_events"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=samples.totals["upper_band"]["num_events"])
        elif feature_key.startswith("hit_"):
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_selected_hits"],
                num_events=samples.totals["lower_band"]["num_events"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_selected_hits"],
                num_events=samples.totals["upper_band"]["num_events"],
            )
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.particle_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_reconstructable_tracks"],
                num_events=samples.totals["lower_band"]["num_events"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.particle_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_reconstructable_tracks"],
                num_events=samples.totals["upper_band"]["num_events"],
            )

        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label=(
                f"lower band (events={samples.totals['lower_band']['num_events']}, "
                f"sample={int(lower_vals.size):,})"
            ),
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label=(
                f"upper band (events={samples.totals['upper_band']['num_events']}, "
                f"sample={int(upper_vals.size):,})"
            ),
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = [counts_lower[2][0].get_label(), counts_upper[2][0].get_label()]

        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_band_mixed_hist_ylabel(feature_key))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    if len(feature_order) < len(axes_flat):
        legend_ax = axes_flat[len(feature_order)]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)

    for ax in axes_flat[len(feature_order) + 1 :]:
        ax.axis("off")
    fig.suptitle(
        f"{samples.split}: band feature comparison "
        f"(hit selection = {samples.hit_selection}, sampled per band)",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_truth_band_comparison(samples: TruthBandSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_true_tracks": "Truth tracks / event",
        "event_true_hits": "True hits / event",
        "event_true_hits_per_true_track": "True hits / truth track",
        "true_hit_x": "True-hit x [m]",
        "true_hit_y": "True-hit y [m]",
        "true_hit_z": "True-hit z [m]",
        "true_hit_r": "True-hit r [m]",
        "true_hit_eta": "True-hit eta",
        "true_hit_phi": "True-hit phi [rad]",
        "true_track_pt": "Truth-track pT [GeV]",
        "true_track_eta": "Truth-track eta",
        "true_track_phi": "Truth-track phi [rad]",
        "true_track_num_hits": "True hits / truth track",
    }
    feature_order = [
        "event_true_tracks",
        "event_true_hits",
        "event_true_hits_per_true_track",
        "true_track_pt",
        "true_track_num_hits",
        "true_track_eta",
        "true_track_phi",
        "true_hit_r",
        "true_hit_eta",
        "true_hit_phi",
        "true_hit_x",
        "true_hit_y",
        "true_hit_z",
    ]

    per_band_features = {band_name: _truth_feature_map(samples, band_name) for band_name in BAND_NAMES}
    fig, axes = plt.subplots(4, 4, figsize=(15.4, 12.8), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    axes_flat = axes.ravel()

    legend_handles = []
    legend_labels = []
    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key in {"true_track_pt"}:
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif feature_key in {"true_track_num_hits", "event_true_tracks", "event_true_hits"}:
            bins, xscale = _count_bins(combined), "linear"
        elif feature_key in {"true_hit_phi", "true_track_phi"}:
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"

        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=samples.totals["lower_band"]["num_events"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=samples.totals["upper_band"]["num_events"])
        elif feature_key.startswith("true_hit_"):
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.hit_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_true_hits"],
                num_events=samples.totals["lower_band"]["num_events"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.hit_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_true_hits"],
                num_events=samples.totals["upper_band"]["num_events"],
            )
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.track_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_true_tracks"],
                num_events=samples.totals["lower_band"]["num_events"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.track_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_true_tracks"],
                num_events=samples.totals["upper_band"]["num_events"],
            )

        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label=(
                f"lower band (events={samples.totals['lower_band']['num_events']}, "
                f"sample={int(lower_vals.size):,})"
            ),
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label=(
                f"upper band (events={samples.totals['upper_band']['num_events']}, "
                f"sample={int(upper_vals.size):,})"
            ),
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = [counts_lower[2][0].get_label(), counts_upper[2][0].get_label()]

        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_band_mixed_hist_ylabel(feature_key))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    if len(feature_order) < len(axes_flat):
        legend_ax = axes_flat[len(feature_order)]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)

    for ax in axes_flat[len(feature_order) + 1 :]:
        ax.axis("off")
    fig.suptitle(
        f"{samples.split}: truth-population comparison across post-filter bands",
        fontsize=13,
        y=0.975,
    )
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_track_loss_comparison(samples: BandTrackLossSamples, output_base: Path) -> None:
    plt = _get_pyplot()
    feature_labels = {
        "event_lost_tracks": "Lost truth tracks / event",
        "event_lost_track_fraction": "Lost-track fraction / event",
        "event_mean_hits_lost_per_truth_track": "Mean hits lost / truth track / event",
        "event_mean_hit_retention_per_truth_track": "Mean hit retention / truth track / event",
        "track_num_hits_lost": "Hits lost / truth track",
        "track_hit_retention": "Hit retention / truth track",
        "track_num_hits_pre": "Truth-track hits before filtering",
        "track_num_hits_post": "Truth-track hits after filtering",
        "track_pt": "Truth-track pT [GeV]",
        "track_eta": "Truth-track eta",
    }
    feature_order = [
        "event_lost_tracks",
        "event_lost_track_fraction",
        "event_mean_hits_lost_per_truth_track",
        "event_mean_hit_retention_per_truth_track",
        "track_num_hits_lost",
        "track_hit_retention",
        "track_num_hits_pre",
        "track_num_hits_post",
        "track_pt",
        "track_eta",
    ]

    per_band_features = {band_name: _track_loss_feature_map(samples, band_name) for band_name in BAND_NAMES}
    fig, axes = plt.subplots(3, 4, figsize=(15.4, 10.2), constrained_layout=False)
    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.30)
    axes_flat = axes.ravel()

    legend_handles = []
    legend_labels = []
    for ax, feature_key in zip(axes_flat, feature_order):
        lower_vals = per_band_features["lower_band"][feature_key]
        upper_vals = per_band_features["upper_band"][feature_key]
        lower_vals = lower_vals[np.isfinite(lower_vals)]
        upper_vals = upper_vals[np.isfinite(upper_vals)]
        combined = _combined_values(lower_vals, upper_vals)
        if feature_key == "track_pt":
            bins, xscale = _feature_bins_and_scale("particle_pt", combined)
        elif feature_key in {"track_eta"}:
            bins, xscale = _continuous_bins(combined), "linear"
        elif "fraction" in feature_key or "retention" in feature_key:
            bins, xscale = np.linspace(0.0, 1.0, 51), "linear"
        else:
            bins, xscale = _count_bins(combined), "linear"

        if feature_key.startswith("event_"):
            lower_weights = _band_hist_weights(len(lower_vals), num_events=samples.totals["lower_band"]["num_events"])
            upper_weights = _band_hist_weights(len(upper_vals), num_events=samples.totals["upper_band"]["num_events"])
        else:
            lower_weights = _sampled_population_hist_weights(
                len(lower_vals),
                sampled_population_size=len(samples.track_samples["lower_band"]),
                total_population=samples.totals["lower_band"]["num_truth_tracks"],
                num_events=samples.totals["lower_band"]["num_events"],
            )
            upper_weights = _sampled_population_hist_weights(
                len(upper_vals),
                sampled_population_size=len(samples.track_samples["upper_band"]),
                total_population=samples.totals["upper_band"]["num_truth_tracks"],
                num_events=samples.totals["upper_band"]["num_events"],
            )

        counts_lower = ax.hist(
            lower_vals,
            bins=bins,
            weights=lower_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["lower_band"],
            label=f"lower band (tracks={samples.totals['lower_band']['num_truth_tracks']:,})",
        )
        counts_upper = ax.hist(
            upper_vals,
            bins=bins,
            weights=upper_weights,
            density=_hist_density(),
            histtype="step",
            linewidth=1.5,
            color=BAND_COLORS["upper_band"],
            label=f"upper band (tracks={samples.totals['upper_band']['num_truth_tracks']:,})",
        )
        if not legend_handles:
            legend_handles = [counts_lower[2][0], counts_upper[2][0]]
            legend_labels = [counts_lower[2][0].get_label(), counts_upper[2][0].get_label()]

        ax.set_title(feature_labels[feature_key], fontsize=11)
        ax.set_ylabel(_band_mixed_hist_ylabel(feature_key))
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    if len(feature_order) < len(axes_flat):
        legend_ax = axes_flat[len(feature_order)]
        legend_ax.axis("off")
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)

    for ax in axes_flat[len(feature_order) + 1 :]:
        ax.axis("off")
    fig.suptitle(f"{samples.split}: reconstructable-track hit-loss comparison across post-filter bands", fontsize=13, y=0.975)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_band_analysis(
    cfg: dict,
    split_counts: SplitCounts,
    output_dir: Path,
    output_stem: str,
    hit_selection: str,
    band_clustering: str,
    max_hit_samples_per_band: int,
    max_particle_samples_per_band: int,
    max_score_samples_per_band: int,
    rng_seed: int,
    gap_half_width: float,
    run_band_explainer: bool,
    band_explainer_hit_bins: int,
    band_explainer_top_features: int,
    band_explainer_test_fraction: float,
    threshold_sweep_points: int,
    crowding_distance: float,
    top_region_keys: int,
    saved_output_dir: Path | None,
    save_band_caches: bool,
    skip_if_not_cached: bool = False,
    plot_sections: set[str] | None = None,
) -> None:
    if plot_sections is None:
        plot_sections = set()

    def _want(section: str) -> bool:
        return not plot_sections or section in plot_sections

    assignments = _assign_event_bands(split_counts, band_clustering=band_clustering)
    assignment_base = output_dir / f"{output_stem}_{split_counts.split}_band_assignments"
    scatter_base = output_dir / f"{output_stem}_{split_counts.split}_band_scatter"
    feature_base = output_dir / f"{output_stem}_{split_counts.split}_band_feature_comparison"
    truth_feature_base = output_dir / f"{output_stem}_{split_counts.split}_band_truth_feature_comparison"
    track_loss_base = output_dir / f"{output_stem}_{split_counts.split}_band_track_loss_comparison"
    cause_diag_base = output_dir / f"{output_stem}_{split_counts.split}_band_cause_diagnostics"
    volume_diag_base = output_dir / f"{output_stem}_{split_counts.split}_band_volume_diagnostics"
    explainer_base = output_dir / f"{output_stem}_{split_counts.split}_band_explainer"
    score_diag_base = output_dir / f"{output_stem}_{split_counts.split}_band_score_diagnostics"
    threshold_sweep_base = output_dir / f"{output_stem}_{split_counts.split}_band_threshold_sweep"
    region_diag_base = output_dir / f"{output_stem}_{split_counts.split}_band_region_conditioned_diagnostics"
    invalid_source_base = output_dir / f"{output_stem}_{split_counts.split}_band_nonreco_source_comparison"
    noise_source_base = output_dir / f"{output_stem}_{split_counts.split}_band_noise_hit_comparison"
    fail_breakdown_base = output_dir / f"{output_stem}_{split_counts.split}_band_nonreco_failure_breakdown"
    reason_conditioned_base = output_dir / f"{output_stem}_{split_counts.split}_band_nonreco_reason_conditioned"
    prefilter_lowpt_noise_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_noise_comparison"
    prefilter_lowpt_noise_status_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_noise_filter_status"
    prefilter_lowpt_event_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_event_diagnostics"
    prefilter_lowpt_proximity_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_proximity_diagnostics"
    prefilter_lowpt_region_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_region_diagnostics"

    if _want("assignments"):
        _write_band_assignments_csv(assignment_base.with_suffix(".csv"), assignments)
    if _want("scatter"):
        _plot_band_scatter(assignments, scatter_base)

    event_diagnostics = None
    volume_ids: list[int] = []
    if _want("cause_diagnostics") or _want("volume_diagnostics") or (_want("explainer") and run_band_explainer):
        cached_event_diagnostics = _maybe_load_saved_event_diagnostics(saved_output_dir, output_stem, split_counts.split)
        if cached_event_diagnostics is None and skip_if_not_cached:
            print(f"[{split_counts.split}] skipping event_diagnostics: not found in cache")
        elif cached_event_diagnostics is None:
            event_diagnostics, volume_ids = _collect_event_cause_diagnostics(
                cfg=cfg,
                assignments=assignments,
                gap_half_width=gap_half_width,
                crowding_distance=crowding_distance,
            )
        else:
            event_diagnostics, volume_ids = cached_event_diagnostics
        if event_diagnostics is not None:
            _write_event_cause_diagnostics_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_event_diagnostics.csv",
                event_diagnostics,
            )
            _write_event_cause_summary_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_event_diagnostics_summary.csv",
                event_diagnostics,
            )
            if _want("cause_diagnostics"):
                _plot_event_cause_diagnostics(event_diagnostics, cause_diag_base, gap_half_width=gap_half_width)
            if _want("volume_diagnostics") and volume_ids:
                _plot_volume_fraction_diagnostics(event_diagnostics, volume_ids=volume_ids, output_base=volume_diag_base)

    hitfilter_diagnostics = None
    if _want("score_diagnostics") or _want("threshold_sweep") or _want("region_diagnostics"):
        hitfilter_diagnostics = _maybe_load_saved_hitfilter_diagnostics(saved_output_dir, output_stem, split_counts.split)
        if hitfilter_diagnostics is None and skip_if_not_cached:
            print(f"[{split_counts.split}] skipping hitfilter_diagnostics: not found in cache")
        elif hitfilter_diagnostics is None:
            hitfilter_diagnostics = _collect_band_hitfilter_diagnostics(
                cfg=cfg,
                assignments=assignments,
                max_score_samples_per_band=max_score_samples_per_band,
                rng_seed=rng_seed,
                num_threshold_points=threshold_sweep_points,
                crowding_distance=crowding_distance,
            )
    if hitfilter_diagnostics is not None:
        hitfilter_diagnostics.threshold_sweep.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_threshold_sweep.csv",
            index=False,
        )
        hitfilter_diagnostics.region_summary.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_region_conditioned_summary.csv",
            index=False,
        )
        hitfilter_diagnostics.invalid_category_summary.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_invalid_category_summary.csv",
            index=False,
        )
        hitfilter_diagnostics.score_samples.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_score_samples.csv",
            index=False,
        )
        _write_band_score_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_score_summary.csv",
            hitfilter_diagnostics.score_samples,
        )
        if _want("score_diagnostics"):
            _plot_band_score_diagnostics(
                hitfilter_diagnostics,
                output_base=score_diag_base,
                current_threshold=float(cfg["data"].get("hit_filter_threshold", 0.1)),
            )
        if _want("threshold_sweep"):
            _plot_band_threshold_sweep(hitfilter_diagnostics, threshold_sweep_base)
        if _want("region_diagnostics"):
            _plot_band_region_conditioned_diagnostics(
                hitfilter_diagnostics,
                output_base=region_diag_base,
                top_region_keys=top_region_keys,
            )

    prefilter_truth_nonreco_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_truth_nonreco_comparison"
    prefilter_truth_nonreco_event_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_truth_nonreco_event_metrics"
    prefilter_truth_noise_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_truth_noise_comparison"
    prefilter_lowpt_kinematics_base = output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_particle_kinematics"
    need_invalid_source_samples = (
        _want("invalid_source_comparison")
        or _want("noise_hit_comparison")
        or _want("nonreco_failure_breakdown")
        or _want("nonreco_reason_conditioned")
    )
    need_prefilter_lowpt_noise_samples = (
        _want("prefilter_lowpt_noise_comparison")
        or _want("prefilter_lowpt_noise_filter_status")
        or _want("prefilter_lowpt_event_diagnostics")
        or _want("prefilter_lowpt_proximity_diagnostics")
        or _want("prefilter_lowpt_region_diagnostics")
    )
    need_prefilter_truth_nonreco_samples = (
        _want("prefilter_truth_nonreco_comparison")
        or _want("prefilter_truth_nonreco_event_metrics")
        or _want("prefilter_truth_noise_comparison")
        or _want("prefilter_lowpt_particle_kinematics")
    )

    invalid_source_samples = None
    if need_invalid_source_samples:
        invalid_source_samples = _maybe_load_pickle_cache(saved_output_dir, output_stem, split_counts.split, "invalid_source_samples")

    prefilter_lowpt_noise_samples = None
    if need_prefilter_lowpt_noise_samples:
        prefilter_lowpt_noise_samples = _maybe_load_pickle_cache(
            saved_output_dir,
            output_stem,
            split_counts.split,
            "prefilter_lowpt_noise_samples",
        )

    prefilter_truth_nonreco_samples = None
    if need_prefilter_truth_nonreco_samples:
        prefilter_truth_nonreco_samples = _maybe_load_pickle_cache(
            saved_output_dir,
            output_stem,
            split_counts.split,
            "prefilter_truth_nonreco_samples",
        )
    if (
        prefilter_truth_nonreco_samples is not None
        and any("num_fail_contains_pt" not in prefilter_truth_nonreco_samples.totals[band_name] for band_name in BAND_NAMES)
    ):
        print(
            f"[{split_counts.split}] ignoring cached prefilter_truth_nonreco_samples: "
            "missing num_fail_contains_pt needed for exact count-per-event reweighting"
        )
        prefilter_truth_nonreco_samples = None

    missing_invalid_source_samples = need_invalid_source_samples and invalid_source_samples is None
    missing_prefilter_lowpt_noise_samples = need_prefilter_lowpt_noise_samples and prefilter_lowpt_noise_samples is None
    missing_prefilter_truth_nonreco_samples = need_prefilter_truth_nonreco_samples and prefilter_truth_nonreco_samples is None

    if missing_invalid_source_samples or missing_prefilter_lowpt_noise_samples or missing_prefilter_truth_nonreco_samples:
        if skip_if_not_cached:
            if missing_invalid_source_samples:
                print(f"[{split_counts.split}] skipping invalid_source_samples: not found in cache")
            if missing_prefilter_lowpt_noise_samples:
                print(f"[{split_counts.split}] skipping prefilter_lowpt_noise_samples: not found in cache")
            if missing_prefilter_truth_nonreco_samples:
                print(f"[{split_counts.split}] skipping prefilter_truth_nonreco_samples: not found in cache")
        else:
            (
                bundled_invalid_source_samples,
                bundled_prefilter_lowpt_noise_samples,
                bundled_prefilter_truth_nonreco_samples,
            ) = _collect_band_raw_event_sample_bundle(
                cfg=cfg,
                assignments=assignments,
                collect_invalid_source=missing_invalid_source_samples,
                collect_prefilter_lowpt_noise=missing_prefilter_lowpt_noise_samples,
                collect_prefilter_truth_nonreco=missing_prefilter_truth_nonreco_samples,
                max_hit_samples_per_band=max_hit_samples_per_band,
                max_particle_samples_per_band=max_particle_samples_per_band,
                rng_seed=rng_seed,
                crowding_distance=crowding_distance,
            )
            if missing_invalid_source_samples:
                invalid_source_samples = bundled_invalid_source_samples
            if missing_prefilter_lowpt_noise_samples:
                prefilter_lowpt_noise_samples = bundled_prefilter_lowpt_noise_samples
            if missing_prefilter_truth_nonreco_samples:
                prefilter_truth_nonreco_samples = bundled_prefilter_truth_nonreco_samples

    if invalid_source_samples is not None:
        if save_band_caches:
            _save_pickle_cache(
                _band_cache_path(output_dir, output_stem, split_counts.split, "invalid_source_samples"),
                invalid_source_samples,
            )
        _write_invalid_source_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_invalid_source_summary.csv",
            invalid_source_samples,
        )
        invalid_source_samples.failure_summary.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_nonreco_failure_summary.csv",
            index=False,
        )
        _write_nonreco_reason_feature_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_nonreco_reason_feature_summary.csv",
            invalid_source_samples,
        )
        if _want("invalid_source_comparison"):
            _plot_invalid_source_comparison(invalid_source_samples, invalid_source_base)
        if _want("noise_hit_comparison"):
            _plot_noise_hit_comparison(invalid_source_samples, noise_source_base)
        if _want("nonreco_failure_breakdown"):
            _plot_nonreco_failure_breakdown(invalid_source_samples, fail_breakdown_base)
        if _want("nonreco_reason_conditioned"):
            _plot_nonreco_reason_conditioned_comparison(invalid_source_samples, reason_conditioned_base)

    if prefilter_lowpt_noise_samples is not None:
        if save_band_caches:
            _save_pickle_cache(
                _band_cache_path(output_dir, output_stem, split_counts.split, "prefilter_lowpt_noise_samples"),
                prefilter_lowpt_noise_samples,
            )
        _write_prefilter_lowpt_noise_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_noise_summary.csv",
            prefilter_lowpt_noise_samples,
        )
        prefilter_lowpt_noise_samples.event_diagnostics.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_event_diagnostics.csv",
            index=False,
        )
        prefilter_lowpt_noise_samples.region_summary.to_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_prefilter_lowpt_region_summary.csv",
            index=False,
        )
        if _want("prefilter_lowpt_noise_comparison"):
            _plot_prefilter_lowpt_noise_comparison(prefilter_lowpt_noise_samples, prefilter_lowpt_noise_base)
        if _want("prefilter_lowpt_noise_filter_status"):
            _plot_prefilter_lowpt_noise_filter_status(prefilter_lowpt_noise_samples, prefilter_lowpt_noise_status_base)
        if _want("prefilter_lowpt_event_diagnostics"):
            _plot_prefilter_lowpt_event_diagnostics(prefilter_lowpt_noise_samples, prefilter_lowpt_event_base)
        if _want("prefilter_lowpt_proximity_diagnostics"):
            _plot_prefilter_lowpt_proximity_comparison(prefilter_lowpt_noise_samples, prefilter_lowpt_proximity_base)
        if _want("prefilter_lowpt_region_diagnostics"):
            _plot_prefilter_lowpt_region_diagnostics(prefilter_lowpt_noise_samples, prefilter_lowpt_region_base, top_region_keys=top_region_keys)

    if prefilter_truth_nonreco_samples is not None:
        if save_band_caches:
            _save_pickle_cache(
                _band_cache_path(output_dir, output_stem, split_counts.split, "prefilter_truth_nonreco_samples"),
                prefilter_truth_nonreco_samples,
            )
        if _want("prefilter_truth_nonreco_comparison"):
            _plot_prefilter_truth_nonreco_comparison(prefilter_truth_nonreco_samples, prefilter_truth_nonreco_base)
        if _want("prefilter_truth_nonreco_event_metrics"):
            _plot_prefilter_truth_nonreco_event_metrics(prefilter_truth_nonreco_samples, prefilter_truth_nonreco_event_base)
        if _want("prefilter_truth_noise_comparison"):
            _plot_prefilter_truth_noise_comparison(prefilter_truth_nonreco_samples, prefilter_truth_noise_base)
        if _want("prefilter_lowpt_particle_kinematics"):
            _plot_prefilter_lowpt_particle_kinematics(prefilter_truth_nonreco_samples, prefilter_lowpt_kinematics_base)

    if run_band_explainer and _want("explainer") and event_diagnostics is not None:
        explainer_results = _maybe_load_saved_band_explainer(saved_output_dir, output_stem, split_counts.split)
        if explainer_results is None and skip_if_not_cached:
            print(f"[{split_counts.split}] skipping band_explainer: not found in cache")
            explainer_results = None
        elif explainer_results is None:
            explainer_results = _fit_band_explainer(
                diagnostics=event_diagnostics,
                split=split_counts.split,
                num_hit_bins=band_explainer_hit_bins,
                test_fraction=band_explainer_test_fraction,
            )
        if explainer_results is not None:
            explainer_results.coefficients.to_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_explainer_coefficients.csv",
                index=False,
            )
            explainer_results.predictions.to_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_explainer_predictions.csv",
                index=False,
            )
            pd.concat(
                [
                    explainer_results.predictions.loc[:, ["event_index", "sample_id", "event_name", "band", "position_group", "band_position"]],
                    explainer_results.residualized_features,
                ],
                axis=1,
            ).to_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_explainer_residualized_features.csv",
                index=False,
            )
            _write_band_explainer_summary_csv(
                output_dir / f"{output_stem}_{split_counts.split}_band_explainer_summary.csv",
                explainer_results,
            )
            _plot_band_explainer(
                diagnostics=event_diagnostics,
                results=explainer_results,
                output_base=explainer_base,
                top_features=band_explainer_top_features,
            )
            print(
                f"[{split_counts.split}] band explainer: "
                f"test ROC AUC={explainer_results.summary['test_roc_auc']:.3f}, "
                f"features={int(explainer_results.summary['num_features'])}"
            )

    need_truth_samples = _want("truth_feature_comparison")
    need_track_loss_samples = _want("track_loss_comparison")
    need_feature_samples = _want("feature_comparison")
    truth_samples = _maybe_load_pickle_cache(saved_output_dir, output_stem, split_counts.split, "truth_samples") if need_truth_samples else None
    track_loss_samples = _maybe_load_pickle_cache(saved_output_dir, output_stem, split_counts.split, "track_loss_samples") if need_track_loss_samples else None
    samples = _maybe_load_pickle_cache(saved_output_dir, output_stem, split_counts.split, "feature_samples") if need_feature_samples else None
    _need_combined = (
        (need_truth_samples and truth_samples is None)
        or (need_track_loss_samples and track_loss_samples is None)
        or (need_feature_samples and samples is None)
    )
    if _need_combined and skip_if_not_cached:
        missing = [
            name
            for name, needed, val in [
                ("truth_samples", need_truth_samples, truth_samples),
                ("track_loss_samples", need_track_loss_samples, track_loss_samples),
                ("feature_samples", need_feature_samples, samples),
            ]
            if needed and val is None
        ]
        print(f"[{split_counts.split}] skipping {', '.join(missing)}: not found in cache")
    elif _need_combined:
        combined_truth_samples, combined_track_loss_samples, combined_feature_samples = _collect_band_truth_trackloss_feature_samples(
            cfg=cfg,
            assignments=assignments,
            hit_selection=hit_selection,
            max_hit_samples_per_band=max_hit_samples_per_band,
            max_particle_samples_per_band=max_particle_samples_per_band,
            rng_seed=rng_seed,
        )
        if need_truth_samples and truth_samples is None:
            truth_samples = combined_truth_samples
        if need_track_loss_samples and track_loss_samples is None:
            track_loss_samples = combined_track_loss_samples
        if need_feature_samples and samples is None:
            samples = combined_feature_samples
    if save_band_caches:
        if truth_samples is not None:
            _save_pickle_cache(_band_cache_path(output_dir, output_stem, split_counts.split, "truth_samples"), truth_samples)
        if track_loss_samples is not None:
            _save_pickle_cache(
                _band_cache_path(output_dir, output_stem, split_counts.split, "track_loss_samples"),
                track_loss_samples,
            )
        if samples is not None:
            _save_pickle_cache(_band_cache_path(output_dir, output_stem, split_counts.split, "feature_samples"), samples)

    if _want("truth_feature_comparison") and truth_samples is not None:
        _write_truth_band_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_truth_feature_summary.csv",
            truth_samples,
        )
        _plot_truth_band_comparison(truth_samples, truth_feature_base)

    if _want("track_loss_comparison") and track_loss_samples is not None:
        _write_track_loss_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_track_loss_summary.csv",
            track_loss_samples,
        )
        _plot_track_loss_comparison(track_loss_samples, track_loss_base)

    if _want("feature_comparison") and samples is not None:
        _write_band_feature_summary_csv(
            output_dir / f"{output_stem}_{split_counts.split}_band_feature_summary.csv",
            samples=samples,
        )
        _plot_band_feature_comparison(samples, feature_base)

    band_counts = {
        band_name: int(np.sum(assignments.band_codes == code))
        for code, band_name in enumerate(BAND_NAMES)
    }
    print(
        f"[{split_counts.split}] band analysis complete: "
        f"{BAND_NAMES[0]}={band_counts[BAND_NAMES[0]]}, {BAND_NAMES[1]}={band_counts[BAND_NAMES[1]]}, "
        f"centers={assignments.band_centers.tolist()}"
    )


def _write_per_event_csv(
    out_path: Path,
    counts_by_split: list[SplitCounts],
    truth_counts_by_split: dict[str, SplitCounts] | None,
    hit_count_column: str,
) -> None:
    include_pre_filter_hits = bool(truth_counts_by_split) and all(
        truth_counts.pre_filter_hits is not None for truth_counts in truth_counts_by_split.values()
    )
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["split", "event_index", "sample_id", "event_name", hit_count_column, "num_reconstructable_tracks"]
        if truth_counts_by_split is not None:
            if include_pre_filter_hits:
                header.append("num_hits_before_filter")
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
                    if include_pre_filter_hits:
                        row.append(int(truth_counts.pre_filter_hits[idx]))
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


def _band_cache_path(base_dir: Path, output_stem: str, split: str, cache_name: str) -> Path:
    return base_dir / f"{output_stem}_{split}_band_{cache_name}.pkl"


def _save_pickle_cache(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle_cache(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


def _maybe_load_pickle_cache(base_dir: Path | None, output_stem: str, split: str, cache_name: str) -> object | None:
    if base_dir is None:
        return None
    path = _band_cache_path(base_dir, output_stem, split, cache_name)
    if not path.exists():
        return None
    print(f"[{split}] loading cached {cache_name.replace('_', ' ')} from {path}")
    return _load_pickle_cache(path)


def _load_fit_summary_metadata(path: Path) -> dict[str, dict[str, Path | None]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    metadata: dict[str, dict[str, Path | None]] = {}
    for _, row in df.iterrows():
        raw_hit_eval = row.get("hit_eval_path")
        hit_eval_path = None if pd.isna(raw_hit_eval) or str(raw_hit_eval) in {"", "None"} else Path(str(raw_hit_eval))
        metadata[str(row["split"])] = {
            "split_dir": Path(str(row["split_dir"])),
            "hit_eval_path": hit_eval_path,
        }
    return metadata


def _load_saved_counts(
    saved_output_dir: Path,
    output_stem: str,
    splits: list[str],
    max_events: int,
) -> tuple[list[SplitCounts], list[SplitCounts]]:
    per_event_path = saved_output_dir / f"{output_stem}_per_event.csv"
    if not per_event_path.exists():
        raise FileNotFoundError(f"Saved per-event table does not exist: {per_event_path}")

    per_event = pd.read_csv(per_event_path)
    fit_meta = _load_fit_summary_metadata(saved_output_dir / f"{output_stem}_fit_summary.csv")
    truth_fit_meta = _load_fit_summary_metadata(saved_output_dir / f"{output_stem}_truth_fit_summary.csv")
    counts_by_split: list[SplitCounts] = []
    truth_counts_by_split: list[SplitCounts] = []

    for split in splits:
        split_df = per_event.loc[per_event["split"] == split].sort_values("event_index").reset_index(drop=True)
        if split_df.empty:
            raise RuntimeError(f"Saved per-event table does not contain split '{split}'.")
        if max_events >= 0:
            split_df = split_df.iloc[:max_events].copy()

        meta = fit_meta.get(split) or truth_fit_meta.get(split)
        if meta is None:
            raise RuntimeError(f"Missing fit-summary metadata for split '{split}' in {saved_output_dir}.")

        hit_column = "num_hits_after_filter" if "num_hits_after_filter" in split_df.columns else "num_hits"
        pre_filter_hits = (
            split_df["num_hits_before_filter"].to_numpy(dtype=np.int32)
            if "num_hits_before_filter" in split_df.columns
            else None
        )
        event_names = split_df["event_name"].astype(str).tolist()
        sample_ids = split_df["sample_id"].to_numpy(dtype=np.int32).tolist()
        split_dir = meta["split_dir"]
        hit_eval_path = meta["hit_eval_path"]

        counts_by_split.append(
            SplitCounts(
                split=split,
                event_names=event_names,
                sample_ids=sample_ids,
                hits=split_df[hit_column].to_numpy(dtype=np.int32),
                particles=split_df["num_reconstructable_tracks"].to_numpy(dtype=np.int32),
                split_dir=split_dir,
                hit_eval_path=hit_eval_path,
                pre_filter_hits=pre_filter_hits,
            )
        )

        if not {"num_truth_hits_before_filter", "num_truth_reconstructable_tracks"}.issubset(split_df.columns):
            raise RuntimeError(
                f"Saved per-event table {per_event_path} is missing truth columns required for split '{split}'."
            )
        truth_counts_by_split.append(
            SplitCounts(
                split=split,
                event_names=event_names,
                sample_ids=sample_ids,
                hits=split_df["num_truth_hits_before_filter"].to_numpy(dtype=np.int32),
                particles=split_df["num_truth_reconstructable_tracks"].to_numpy(dtype=np.int32),
                split_dir=split_dir,
                hit_eval_path=None,
                pre_filter_hits=pre_filter_hits,
            )
        )

    return counts_by_split, truth_counts_by_split


def _infer_volume_ids_from_event_diagnostics(diagnostics: pd.DataFrame) -> list[int]:
    prefix = "hit_fraction_volume_"
    volume_ids = [int(col[len(prefix) :]) for col in diagnostics.columns if col.startswith(prefix)]
    return sorted(volume_ids)


def _maybe_load_saved_event_diagnostics(
    saved_output_dir: Path | None,
    output_stem: str,
    split: str,
) -> tuple[pd.DataFrame, list[int]] | None:
    if saved_output_dir is None:
        return None
    path = saved_output_dir / f"{output_stem}_{split}_band_event_diagnostics.csv"
    if not path.exists():
        return None
    diagnostics = pd.read_csv(path)
    print(f"[{split}] loading saved event diagnostics from {path}")
    return diagnostics, _infer_volume_ids_from_event_diagnostics(diagnostics)


def _maybe_load_saved_hitfilter_diagnostics(
    saved_output_dir: Path | None,
    output_stem: str,
    split: str,
) -> BandHitFilterDiagnostics | None:
    if saved_output_dir is None:
        return None
    paths = {
        "score_samples": saved_output_dir / f"{output_stem}_{split}_band_score_samples.csv",
        "threshold_sweep": saved_output_dir / f"{output_stem}_{split}_band_threshold_sweep.csv",
        "region_summary": saved_output_dir / f"{output_stem}_{split}_band_region_conditioned_summary.csv",
        "invalid_category_summary": saved_output_dir / f"{output_stem}_{split}_band_invalid_category_summary.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    invalid_category_summary = pd.read_csv(paths["invalid_category_summary"])
    if "total_invalid_in_category" not in invalid_category_summary.columns:
        print(
            f"[{split}] ignoring saved hit-filter diagnostics from {saved_output_dir}: "
            "missing total_invalid_in_category needed for exact count-per-event reweighting"
        )
        return None
    print(f"[{split}] loading saved hit-filter diagnostics from {saved_output_dir}")
    return BandHitFilterDiagnostics(
        split=split,
        score_samples=pd.read_csv(paths["score_samples"]),
        threshold_sweep=pd.read_csv(paths["threshold_sweep"]),
        region_summary=pd.read_csv(paths["region_summary"]),
        invalid_category_summary=invalid_category_summary,
    )


def _maybe_load_saved_band_explainer(
    saved_output_dir: Path | None,
    output_stem: str,
    split: str,
) -> BandExplainerResults | None:
    if saved_output_dir is None:
        return None
    coef_path = saved_output_dir / f"{output_stem}_{split}_band_explainer_coefficients.csv"
    pred_path = saved_output_dir / f"{output_stem}_{split}_band_explainer_predictions.csv"
    resid_path = saved_output_dir / f"{output_stem}_{split}_band_explainer_residualized_features.csv"
    summary_path = saved_output_dir / f"{output_stem}_{split}_band_explainer_summary.csv"
    if not all(path.exists() for path in (coef_path, pred_path, resid_path, summary_path)):
        return None

    summary_df = pd.read_csv(summary_path)
    summary: dict[str, float] = {}
    split_name = split
    for _, row in summary_df.iterrows():
        metric = str(row["metric"])
        if metric == "split":
            split_name = str(row["value"])
        else:
            summary[metric] = float(row["value"])

    residualized = pd.read_csv(resid_path)
    residualized = residualized.drop(
        columns=["event_index", "sample_id", "event_name", "band", "position_group", "band_position"],
        errors="ignore",
    )
    coefficients = pd.read_csv(coef_path)
    feature_columns = coefficients["feature"].astype(str).tolist() if "feature" in coefficients.columns else residualized.columns.tolist()
    print(f"[{split}] loading saved band explainer results from {saved_output_dir}")
    return BandExplainerResults(
        split=split_name,
        feature_columns=feature_columns,
        residualized_features=residualized,
        predictions=pd.read_csv(pred_path),
        coefficients=coefficients,
        summary=summary,
    )


def main() -> None:
    global PLOT_HIST_MODE

    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()
    if args.hist_mode is not None:
        PLOT_HIST_MODE = str(args.hist_mode)
    else:
        PLOT_HIST_MODE = "density" if bool(args.hist_density) else "count"
    if args.powerpoint_only:
        args.analyze_bands = True
    band_plot_sections = set(POWERPOINT_BAND_PLOT_SECTIONS) if args.powerpoint_only else set()

    cfg = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_from_output_dir is not None:
        counts_by_split, truth_counts_by_split_list = _load_saved_counts(
            saved_output_dir=args.load_from_output_dir,
            output_stem=args.output_stem,
            splits=args.splits,
            max_events=args.max_events,
        )
    else:
        counts_by_split = [
            _collect_counts(
                cfg=cfg,
                split=split,
                max_events=args.max_events,
                use_hit_eval=args.use_hit_eval,
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
    truth_counts_by_split = _validate_matching_events(counts_by_split, truth_counts_by_split_list)

    has_hit_eval = all(split_counts.hit_eval_path is not None for split_counts in counts_by_split)
    x_axis_label = "Hits after hit filtering" if has_hit_eval else "Hits per event"
    hit_count_column = "num_hits_after_filter" if has_hit_eval else "num_hits"
    hit_mean_column = "mean_hits_after_filter" if has_hit_eval else "mean_hits"

    # Derive pre-filter SplitCounts from the truth pass (same raw data, no extra I/O).
    pre_filter_counts_by_split = None
    if all(sc.pre_filter_hits is not None for sc in truth_counts_by_split_list):
        pre_filter_counts_by_split = [
            SplitCounts(
                split=sc.split,
                event_names=sc.event_names,
                sample_ids=sc.sample_ids,
                hits=sc.pre_filter_hits,
                particles=sc.particles,
                split_dir=sc.split_dir,
                hit_eval_path=None,
            )
            for sc in truth_counts_by_split_list
        ]
    elif args.load_from_output_dir is not None:
        print(
            "Skipping pre-filter panel: saved per-event table does not contain "
            "'num_hits_before_filter'. Rerun once without --load-from-output-dir to cache it."
        )

    output_base = output_dir / args.output_stem
    _plot_panels(
        counts_by_split=counts_by_split,
        add_linear_fit=args.add_linear_fit,
        output_base=output_base,
        x_axis_label=x_axis_label,
    )
    truth_output_base = output_dir / f"{args.output_stem}_truth"
    _plot_panels(
        counts_by_split=truth_counts_by_split_list,
        add_linear_fit=args.add_linear_fit,
        output_base=truth_output_base,
        x_axis_label="Truth hits on reconstructable tracks",
    )
    pre_filter_output_base = output_dir / f"{args.output_stem}_pre_filter"
    if pre_filter_counts_by_split is not None:
        _plot_panels(
            counts_by_split=pre_filter_counts_by_split,
            add_linear_fit=args.add_linear_fit,
            output_base=pre_filter_output_base,
            x_axis_label="Hits before hit filtering",
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
    _write_fit_summary_csv(
        truth_output_base.with_name(truth_output_base.name + "_fit_summary.csv"),
        counts_by_split=truth_counts_by_split_list,
        hit_mean_column="mean_truth_hits_on_reconstructable_tracks",
    )

    if args.analyze_bands:
        if not args.use_hit_eval:
            raise ValueError("--analyze-bands requires --use-hit-eval because the band artefact is defined on post-filter hits.")
        for split_counts in counts_by_split:
            _run_band_analysis(
                cfg=cfg,
                split_counts=split_counts,
                output_dir=output_dir,
                output_stem=args.output_stem,
                hit_selection=args.band_hit_selection,
                band_clustering=args.band_clustering,
                max_hit_samples_per_band=args.band_max_hit_samples_per_band,
                max_particle_samples_per_band=args.band_max_particle_samples_per_band,
                max_score_samples_per_band=args.band_max_score_samples_per_band,
                rng_seed=args.band_rng_seed,
                gap_half_width=args.band_gap_half_width,
                run_band_explainer=args.band_explainer,
                band_explainer_hit_bins=args.band_explainer_hit_bins,
                band_explainer_top_features=args.band_explainer_top_features,
                band_explainer_test_fraction=args.band_explainer_test_fraction,
                threshold_sweep_points=args.band_threshold_sweep_points,
                crowding_distance=args.band_crowding_distance,
                top_region_keys=args.band_top_region_keys,
                saved_output_dir=args.load_from_output_dir,
                save_band_caches=args.save_band_caches,
                skip_if_not_cached=args.skip_if_not_cached,
                plot_sections=band_plot_sections,
            )

    print(f"Saved figure to {output_base.with_suffix('.pdf')}")
    print(f"Saved figure to {output_base.with_suffix('.png')}")
    print(f"Saved figure to {truth_output_base.with_suffix('.pdf')}")
    print(f"Saved figure to {truth_output_base.with_suffix('.png')}")
    if pre_filter_counts_by_split is not None:
        print(f"Saved figure to {pre_filter_output_base.with_suffix('.pdf')}")
        print(f"Saved figure to {pre_filter_output_base.with_suffix('.png')}")
    print(f"Saved per-event table to {output_base.with_name(output_base.name + '_per_event.csv')}")
    print(f"Saved fit summary to {output_base.with_name(output_base.name + '_fit_summary.csv')}")
    print(f"Saved fit summary to {truth_output_base.with_name(truth_output_base.name + '_fit_summary.csv')}")
    if args.analyze_bands:
        def _print_band_output(section: str, message: str) -> None:
            if not band_plot_sections or section in band_plot_sections:
                print(message)

        for split in args.splits:
            _print_band_output("assignments", f"Saved band assignments to {output_dir / f'{args.output_stem}_{split}_band_assignments.csv'}")
            _print_band_output("scatter", f"Saved band scatter to {output_dir / f'{args.output_stem}_{split}_band_scatter.png'}")
            _print_band_output("cause_diagnostics", f"Saved band cause diagnostics to {output_dir / f'{args.output_stem}_{split}_band_cause_diagnostics.png'}")
            _print_band_output("volume_diagnostics", f"Saved band volume diagnostics to {output_dir / f'{args.output_stem}_{split}_band_volume_diagnostics.png'}")
            _print_band_output("score_diagnostics", f"Saved band score diagnostics to {output_dir / f'{args.output_stem}_{split}_band_score_diagnostics.png'}")
            _print_band_output("threshold_sweep", f"Saved band threshold sweep to {output_dir / f'{args.output_stem}_{split}_band_threshold_sweep.png'}")
            _print_band_output("threshold_sweep", f"Saved band threshold sweep table to {output_dir / f'{args.output_stem}_{split}_band_threshold_sweep.csv'}")
            _print_band_output("region_diagnostics", f"Saved band region-conditioned diagnostics to {output_dir / f'{args.output_stem}_{split}_band_region_conditioned_diagnostics.png'}")
            _print_band_output("region_diagnostics", f"Saved band region-conditioned summary to {output_dir / f'{args.output_stem}_{split}_band_region_conditioned_summary.csv'}")
            _print_band_output("score_diagnostics", f"Saved band invalid category summary to {output_dir / f'{args.output_stem}_{split}_band_invalid_category_summary.csv'}")
            _print_band_output("invalid_source_comparison", f"Saved band invalid-source comparison to {output_dir / f'{args.output_stem}_{split}_band_nonreco_source_comparison.png'}")
            _print_band_output("noise_hit_comparison", f"Saved band noise-hit comparison to {output_dir / f'{args.output_stem}_{split}_band_noise_hit_comparison.png'}")
            _print_band_output("nonreco_failure_breakdown", f"Saved band nonreco failure breakdown to {output_dir / f'{args.output_stem}_{split}_band_nonreco_failure_breakdown.png'}")
            _print_band_output("nonreco_reason_conditioned", f"Saved band nonreco reason-conditioned comparison to {output_dir / f'{args.output_stem}_{split}_band_nonreco_reason_conditioned.png'}")
            _print_band_output("invalid_source_comparison", f"Saved band invalid-source summary to {output_dir / f'{args.output_stem}_{split}_band_invalid_source_summary.csv'}")
            _print_band_output("nonreco_failure_breakdown", f"Saved band nonreco failure summary to {output_dir / f'{args.output_stem}_{split}_band_nonreco_failure_summary.csv'}")
            _print_band_output("nonreco_reason_conditioned", f"Saved band nonreco reason-feature summary to {output_dir / f'{args.output_stem}_{split}_band_nonreco_reason_feature_summary.csv'}")
            _print_band_output("prefilter_lowpt_noise_comparison", f"Saved band prefilter low-pT/noise comparison to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_noise_comparison.png'}")
            _print_band_output("prefilter_lowpt_noise_filter_status", f"Saved band prefilter low-pT/noise filter-status comparison to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_noise_filter_status.png'}")
            _print_band_output("prefilter_lowpt_event_diagnostics", f"Saved band prefilter low-pT event diagnostics to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_event_diagnostics.png'}")
            _print_band_output("prefilter_lowpt_proximity_diagnostics", f"Saved band prefilter low-pT proximity diagnostics to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_proximity_diagnostics.png'}")
            _print_band_output("prefilter_lowpt_region_diagnostics", f"Saved band prefilter low-pT region diagnostics to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_region_diagnostics.png'}")
            _print_band_output("prefilter_lowpt_noise_comparison", f"Saved band prefilter low-pT/noise summary to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_noise_summary.csv'}")
            _print_band_output("prefilter_truth_nonreco_comparison", f"Saved band pre-filter truth nonreco comparison to {output_dir / f'{args.output_stem}_{split}_band_prefilter_truth_nonreco_comparison.png'}")
            _print_band_output("prefilter_truth_nonreco_event_metrics", f"Saved band pre-filter truth nonreco event metrics to {output_dir / f'{args.output_stem}_{split}_band_prefilter_truth_nonreco_event_metrics.png'}")
            _print_band_output("prefilter_truth_noise_comparison", f"Saved band pre-filter truth noise comparison to {output_dir / f'{args.output_stem}_{split}_band_prefilter_truth_noise_comparison.png'}")
            _print_band_output("prefilter_lowpt_particle_kinematics", f"Saved band pre-filter low-pT particle kinematics to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_particle_kinematics.png'}")
            _print_band_output("prefilter_lowpt_event_diagnostics", f"Saved band prefilter low-pT event diagnostics table to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_event_diagnostics.csv'}")
            _print_band_output("prefilter_lowpt_region_diagnostics", f"Saved band prefilter low-pT region summary to {output_dir / f'{args.output_stem}_{split}_band_prefilter_lowpt_region_summary.csv'}")
            _print_band_output("score_diagnostics", f"Saved band score samples to {output_dir / f'{args.output_stem}_{split}_band_score_samples.csv'}")
            _print_band_output("score_diagnostics", f"Saved band score summary to {output_dir / f'{args.output_stem}_{split}_band_score_summary.csv'}")
            _print_band_output("cause_diagnostics", f"Saved band event diagnostics to {output_dir / f'{args.output_stem}_{split}_band_event_diagnostics.csv'}")
            _print_band_output("cause_diagnostics", f"Saved band event diagnostic summary to {output_dir / f'{args.output_stem}_{split}_band_event_diagnostics_summary.csv'}")
            _print_band_output("truth_feature_comparison", f"Saved band truth-population comparison to {output_dir / f'{args.output_stem}_{split}_band_truth_feature_comparison.png'}")
            _print_band_output("truth_feature_comparison", f"Saved band truth-population summary to {output_dir / f'{args.output_stem}_{split}_band_truth_feature_summary.csv'}")
            _print_band_output("track_loss_comparison", f"Saved band track-loss comparison to {output_dir / f'{args.output_stem}_{split}_band_track_loss_comparison.png'}")
            _print_band_output("track_loss_comparison", f"Saved band track-loss summary to {output_dir / f'{args.output_stem}_{split}_band_track_loss_summary.csv'}")
            if args.band_explainer:
                _print_band_output("explainer", f"Saved band explainer to {output_dir / f'{args.output_stem}_{split}_band_explainer.png'}")
                _print_band_output("explainer", f"Saved band explainer coefficients to {output_dir / f'{args.output_stem}_{split}_band_explainer_coefficients.csv'}")
                _print_band_output("explainer", f"Saved band explainer predictions to {output_dir / f'{args.output_stem}_{split}_band_explainer_predictions.csv'}")
                _print_band_output("explainer", f"Saved band explainer residualized features to {output_dir / f'{args.output_stem}_{split}_band_explainer_residualized_features.csv'}")
                _print_band_output("explainer", f"Saved band explainer summary to {output_dir / f'{args.output_stem}_{split}_band_explainer_summary.csv'}")
            _print_band_output("feature_comparison", f"Saved band feature comparison to {output_dir / f'{args.output_stem}_{split}_band_feature_comparison.png'}")
            _print_band_output("feature_comparison", f"Saved band feature summary to {output_dir / f'{args.output_stem}_{split}_band_feature_summary.csv'}")


if __name__ == "__main__":
    main()
