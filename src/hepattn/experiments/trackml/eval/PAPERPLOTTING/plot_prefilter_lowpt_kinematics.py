#!/usr/bin/env python3
"""
Focused plot: pre-filter sub-threshold-pT particle kinematics by post-filter band.

Produces a 2x2 figure (pT / eta / phi / truth hits per track) comparing lower-band
and upper-band events.  Population: every particle whose pT is at or below
particle_min_pt in the raw (pre-filter) event.

Two input modes
---------------
Fast (recommended)
    --band-assignments-csv  A CSV produced by a prior run of
                            plot_reconstructable_vs_post_filter_hits.py
                            (--analyze-bands writes one per split).
                            No hit-eval file is read; only the raw parquet
                            files are touched.

Self-contained
    --config only           Bands are computed from scratch using the config
                            hit-eval file (one pass) and particle features are
                            collected from the raw parquets (second pass).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Constants matching the main analysis script
# ---------------------------------------------------------------------------
BAND_COLORS = {"lower_band": "#d95f02", "upper_band": "#1b9e77"}
BAND_NAMES = ("lower_band", "upper_band")

DEFAULT_CONFIG = Path(
    "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots/prefilter_lowpt_kinematics/"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot pre-filter sub-threshold-pT particle kinematics by post-filter band.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum events per split (-1 = all).",
    )
    p.add_argument(
        "--band-assignments-csv",
        type=Path,
        default=None,
        help=(
            "Pre-computed band assignments CSV from the main analysis script. "
            "When provided, no hit-eval file is read and bands are not recomputed."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-stem", type=str, default="prefilter_lowpt_kinematics")
    p.add_argument(
        "--max-samples-per-band",
        type=int,
        default=200_000,
        help="Reservoir-sample cap per band for the particle feature arrays.",
    )
    p.add_argument(
        "--hist-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise histograms to density (True) or raw counts (False).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Minimal self-contained helpers
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required. Install it and rerun.") from exc
    with path.open() as f:
        return yaml.safe_load(f)


def _get_pyplot():
    import matplotlib as mpl
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


def _load_raw_event(
    split_dir: Path,
    event_name: str,
    hit_volume_ids: list[int] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    particles = pd.read_parquet(split_dir / f"{event_name}-parts.parquet")
    hits = pd.read_parquet(split_dir / f"{event_name}-hits.parquet")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)].copy()
    else:
        hits = hits.copy()
    particles = particles.copy()
    particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    p_mag = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["eta"] = np.arctanh(particles["pz"] / np.maximum(p_mag, 1e-12))
    particles["phi"] = np.arctan2(particles["py"], particles["px"])
    return hits, particles


def _priority_sample(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    max_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if new_rows.empty:
        return existing
    new_rows = new_rows.copy()
    new_rows["_key"] = rng.random(len(new_rows))
    combined = new_rows if existing.empty else pd.concat([existing, new_rows], ignore_index=True)
    if max_rows > 0 and len(combined) > max_rows:
        combined = combined.nsmallest(max_rows, "_key").reset_index(drop=True)
    return combined


def _combined_values(*arrays: np.ndarray) -> np.ndarray:
    kept = [a[np.isfinite(a)] for a in arrays if a.size > 0]
    return np.concatenate(kept) if kept else np.array([], dtype=np.float64)


def _count_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.arange(-0.5, 1.5, 1.0)
    lo = int(np.floor(np.min(values)))
    hi = int(np.ceil(np.percentile(values, 99.5)))
    hi = max(hi, lo + 1)
    return np.arange(lo - 0.5, hi + 1.5, 1.0) if (hi - lo) <= 60 else np.linspace(lo, hi, 61)


def _continuous_bins(values: np.ndarray, n_bins: int = 60) -> np.ndarray:
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if np.isclose(lo, hi):
        delta = max(1e-6, abs(lo) * 0.05 + 1e-6)
        lo, hi = lo - delta, hi + delta
    return np.linspace(lo, hi, n_bins + 1)


def _pt_bins(values: np.ndarray) -> tuple[np.ndarray, str]:
    pos = values[values > 0]
    if pos.size == 0:
        return np.linspace(0.0, 1.0, 61), "linear"
    lo = max(float(np.percentile(pos, 0.5)), float(np.min(pos)))
    hi = max(float(np.percentile(pos, 99.5)), lo * 1.05)
    return np.geomspace(lo, hi, 61), "log"


# ---------------------------------------------------------------------------
# Band assignment (self-contained path only)
# ---------------------------------------------------------------------------
def _read_hit_filter_scores(hf, sample_id: int) -> np.ndarray | None:
    for path in (
        f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle_prob",
        f"{sample_id}/preds/final/hit_filter/key_on_valid_particle_prob",
        f"{sample_id}/preds/final/hit_filter/hit_on_valid_particle",
    ):
        try:
            scores = np.asarray(hf[path], dtype=np.float64)
            if scores.ndim >= 1 and scores.shape[0] == 1:
                scores = scores[0]
            return scores
        except KeyError:
            continue
    return None


def _median_band_split(residuals: np.ndarray) -> np.ndarray:
    codes = (residuals >= float(np.median(residuals))).astype(np.int32)
    if np.all(codes == codes[0]):
        order = np.argsort(residuals)
        codes = np.ones(residuals.size, dtype=np.int32)
        codes[order[: max(1, residuals.size // 2)]] = 0
    return codes


def _compute_band_codes(
    split: str,
    split_dir: Path,
    event_names: list[str],
    sample_ids: list[int],
    hit_volume_ids: list[int] | None,
    hit_eval_path: Path,
    hit_threshold: float,
) -> np.ndarray:
    """Single pass: read hit-eval scores + raw parquets to get per-event
    (post_filter_hits, particle_count), then split into lower/upper bands."""
    import h5py

    post_hits = np.zeros(len(event_names), dtype=np.int32)
    part_counts = np.zeros(len(event_names), dtype=np.int32)

    print(f"[{split}] computing band assignments over {len(event_names)} events ...")
    with h5py.File(hit_eval_path, "r") as hf:
        for i, (name, sid) in enumerate(zip(event_names, sample_ids)):
            hits_raw, particles_raw = _load_raw_event(split_dir, name, hit_volume_ids)
            scores = _read_hit_filter_scores(hf, sid)
            if scores is None or len(scores) != len(hits_raw):
                raise RuntimeError(
                    f"Missing or misaligned hit-filter scores for sample_id={sid} "
                    f"(event '{name}')."
                )
            post_hits[i] = int(np.sum(scores >= hit_threshold))
            part_counts[i] = len(particles_raw)
            if (i + 1) % 50 == 0 or (i + 1) == len(event_names):
                print(f"[{split}]  {i + 1}/{len(event_names)} events")

    x = post_hits.astype(np.float64)
    y = part_counts.astype(np.float64)
    if x.size < 2 or np.unique(x).size < 2:
        raise RuntimeError(f"Not enough distinct hit counts to fit bands for split '{split}'.")
    slope, intercept = np.polyfit(x, y, deg=1)
    residuals = y - (intercept + slope * x)
    return _median_band_split(residuals)


def _enumerate_split_events(
    split_dir: Path,
    data_cfg: dict,
    split: str,
    max_events: int,
) -> tuple[list[str], list[int]]:
    """Return (event_names, sample_ids) for a split, respecting num_<split> and max_events."""
    parts_files = sorted(glob.glob(str(split_dir / "*-parts.parquet")))
    event_names = [Path(f).name.replace("-parts.parquet", "") for f in parts_files]

    split_num = int(data_cfg.get(f"num_{split}", -1))
    if split_num > 0:
        event_names = event_names[:split_num]
    if max_events > 0:
        event_names = event_names[:max_events]

    # Derive sample_id from the numeric suffix of the event name (e.g. event000021001 → 21001).
    sample_ids = []
    for name in event_names:
        digits = "".join(c for c in name if c.isdigit())
        sample_ids.append(int(digits) if digits else 0)

    return event_names, sample_ids


# ---------------------------------------------------------------------------
# Particle feature collection
# ---------------------------------------------------------------------------
def collect_lowpt_samples(
    split: str,
    split_dir: Path,
    event_names: list[str],
    band_codes: np.ndarray,
    hit_volume_ids: list[int] | None,
    min_pt: float,
    max_samples_per_band: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Return {band_name: DataFrame} with columns pt/eta/phi/num_hits_pre."""
    empty = pd.DataFrame(columns=["pt", "eta", "phi", "num_hits_pre", "_key"])
    samples: dict[str, pd.DataFrame] = {name: empty.copy() for name in BAND_NAMES}

    print(f"[{split}] collecting sub-threshold-pT particle features from {len(event_names)} events ...")
    for i, (event_name, band_code) in enumerate(zip(event_names, band_codes)):
        band_name = BAND_NAMES[int(band_code)]
        hits_raw, particles_raw = _load_raw_event(split_dir, event_name, hit_volume_ids)

        hit_counts = hits_raw["particle_id"].value_counts()
        particles_raw = particles_raw.copy()
        particles_raw["num_hits_pre"] = (
            particles_raw["particle_id"].map(hit_counts).fillna(0).astype(np.int32)
        )

        lowpt = particles_raw.loc[
            particles_raw["pt"].to_numpy(dtype=np.float64) <= min_pt,
            ["pt", "eta", "phi", "num_hits_pre"],
        ].copy()

        samples[band_name] = _priority_sample(samples[band_name], lowpt, max_samples_per_band, rng)

        if (i + 1) % 50 == 0 or (i + 1) == len(event_names):
            print(f"[{split}]  {i + 1}/{len(event_names)} events")

    return samples


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_lowpt_kinematics(
    samples: dict[str, pd.DataFrame],
    split: str,
    output_base: Path,
    hist_density: bool,
) -> None:
    plt = _get_pyplot()

    feature_specs = [
        ("pt",           "pT [GeV]"),
        ("eta",          "eta"),
        ("phi",          "phi [rad]"),
        ("num_hits_pre", "Truth hits per track"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)

    for ax, (feature_key, title) in zip(axes.ravel(), feature_specs):
        lower_vals = samples["lower_band"][feature_key].to_numpy(dtype=np.float64)
        upper_vals = samples["upper_band"][feature_key].to_numpy(dtype=np.float64)
        combined = _combined_values(lower_vals, upper_vals)

        if feature_key == "pt":
            bins, xscale = _pt_bins(combined)
        elif feature_key == "phi":
            bins, xscale = np.linspace(-np.pi, np.pi, 61), "linear"
        elif feature_key == "num_hits_pre":
            bins, xscale = _count_bins(combined), "linear"
        else:
            bins, xscale = _continuous_bins(combined), "linear"

        ax.hist(
            lower_vals, bins=bins, density=hist_density, histtype="step",
            linewidth=1.5, color=BAND_COLORS["lower_band"], label="lower band",
        )
        ax.hist(
            upper_vals, bins=bins, density=hist_density, histtype="step",
            linewidth=1.5, color=BAND_COLORS["upper_band"], label="upper band",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Density" if hist_density else "Count")
        ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        if xscale == "log":
            ax.set_xscale("log")

    axes.ravel()[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{split}: pre-filter sub-threshold-pT particle kinematics by band",
        fontsize=13,
    )
    for suffix in (".pdf", ".png"):
        out = output_base.with_suffix(suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[{split}] saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = _parse_args()

    cfg = _read_yaml(args.config)
    data_cfg = cfg["data"]
    min_pt = float(data_cfg["particle_min_pt"])
    hit_volume_ids = data_cfg.get("hit_volume_ids")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = Path(data_cfg[f"{split}_dir"])
        rng = np.random.default_rng(12345)

        # ------------------------------------------------------------------
        # Step 1: resolve band assignments
        # ------------------------------------------------------------------
        if args.band_assignments_csv is not None:
            band_df = pd.read_csv(args.band_assignments_csv)
            band_df = band_df[band_df["split"] == split].reset_index(drop=True)
            if band_df.empty:
                raise ValueError(
                    f"No rows found for split '{split}' in {args.band_assignments_csv}."
                )
            event_names = band_df["event_name"].tolist()
            band_codes = (band_df["band"] == "upper_band").astype(np.int32).to_numpy()
            if args.max_events > 0:
                event_names = event_names[: args.max_events]
                band_codes = band_codes[: args.max_events]
            print(f"[{split}] loaded {len(event_names)} band assignments from CSV (no hit-eval read)")
        else:
            hit_eval_value = data_cfg.get(f"hit_eval_{split}")
            if not hit_eval_value:
                raise ValueError(
                    f"Config has no data.hit_eval_{split}. "
                    "Either add it to the config or pass --band-assignments-csv."
                )
            hit_eval_path = Path(hit_eval_value)
            hit_threshold = float(data_cfg.get("hit_filter_threshold", 0.1))
            event_names, sample_ids = _enumerate_split_events(
                split_dir, data_cfg, split, args.max_events
            )
            band_codes = _compute_band_codes(
                split=split,
                split_dir=split_dir,
                event_names=event_names,
                sample_ids=sample_ids,
                hit_volume_ids=hit_volume_ids,
                hit_eval_path=hit_eval_path,
                hit_threshold=hit_threshold,
            )

        # ------------------------------------------------------------------
        # Step 2: collect particle features (raw parquets only)
        # ------------------------------------------------------------------
        samples = collect_lowpt_samples(
            split=split,
            split_dir=split_dir,
            event_names=event_names,
            band_codes=band_codes,
            hit_volume_ids=hit_volume_ids,
            min_pt=min_pt,
            max_samples_per_band=args.max_samples_per_band,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # Step 3: plot
        # ------------------------------------------------------------------
        output_base = args.output_dir / f"{args.output_stem}_{split}"
        plot_lowpt_kinematics(samples, split, output_base, hist_density=bool(args.hist_density))


if __name__ == "__main__":
    main()
