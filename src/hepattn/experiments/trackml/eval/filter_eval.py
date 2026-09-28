#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned conversion from Jupyter notebook: filter_output_plots.ipynb
Notebook magics (%, %%, !) are commented out for .py compatibility.
"""



# ------------------------------------------------------------------------------
# Cell 1 (code)
# %load_ext autoreload  # (line magic commented out)
# %autoreload 2  # (line magic commented out)


# ------------------------------------------------------------------------------
# Cell 2 (code)
import argparse
import csv
import math
import pathlib
import sys

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.dpi"] = 400
# plt.rcParams["text.usetex"] = True
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TrackML hit filtering outputs and plot threshold tradeoffs.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Reference threshold highlighted on ROC/efficiency-purity plots.")
    parser.add_argument("--target-efficiency", type=float, default=0.99, help="Target efficiency used to find the anchor threshold.")
    parser.add_argument(
        "--reconstructable-min-hits",
        type=int,
        default=3,
        help="Minimum kept hits required for a particle to count as reconstructable in track-level metrics.",
    )
    parser.add_argument(
        "--run-full-plots",
        action="store_true",
        default=True,
        help="Load full event DataFrames and generate all plots.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap for quick checks.")
    parser.add_argument(
        "--report-threshold-scan",
        action="store_true",
        help="Print metrics for sampled thresholds in an efficiency window around the target efficiency.",
    )
    parser.add_argument(
        "--threshold-scan-points",
        type=int,
        default=8,
        help="Number of sampled threshold points reported for the threshold scan window.",
    )
    parser.add_argument(
        "--threshold-scan-eff-window",
        type=float,
        default=0.005,
        help="Absolute efficiency half-window around target used for threshold scanning (e.g. 0.005 => target +/- 0.5%).",
    )
    parser.add_argument(
        "--threshold-scan-csv",
        type=str,
        default=None,
        help="Optional CSV path to save threshold scan metrics.",
    )
    parser.add_argument(
        "--annotate-threshold-scan",
        action="store_true",
        help="Annotate threshold values next to threshold-scan points on efficiency-purity plots.",
    )
    parser.add_argument(
        "--only-threshold-metrics",
        action="store_true",
        help="Exit after threshold metrics output without generating plots (implied when --run-full-plots is not set).",
    )
    return parser.parse_known_args()[0]


def aligned_eff_purity_thresholds(metrics: dict):
    eff = np.asarray(metrics["roc_eff"], dtype=float)
    pur = np.asarray(metrics["roc_pur"], dtype=float)
    thr = np.asarray(metrics["roc_eff_pur_thr"], dtype=float)
    if eff.size == thr.size + 1:
        eff = eff[:-1]
    if pur.size == thr.size + 1:
        pur = pur[:-1]
    n = min(eff.size, pur.size, thr.size)
    return eff[:n], pur[:n], thr[:n]


def hit_purity_efficiency_curve(metrics: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hit efficiency, hit purity, and thresholds for the PR-style curve.

    Hit purity is TP / (TP + FP): among hits predicted true, which fraction are true.
    Hit efficiency is TP / (TP + FN): among all true hits, which fraction are predicted true.
    """
    return aligned_eff_purity_thresholds(metrics)


def hit_curve_point_for_threshold(metrics: dict, threshold: float) -> tuple[float, float, float]:
    eff, pur, thr = hit_purity_efficiency_curve(metrics)
    idx = int(np.argmin(np.abs(thr - threshold)))
    return float(eff[idx]), float(pur[idx]), float(thr[idx])


def threshold_index_for_efficiency(metrics: dict, target_eff: float = 0.99) -> int:
    eff, _pur, _thr = aligned_eff_purity_thresholds(metrics)
    passing = np.flatnonzero(eff >= target_eff)
    if passing.size:
        return int(passing[-1])
    return int(np.argmin(np.abs(eff - target_eff)))


def threshold_for_efficiency(metrics: dict, target_eff: float = 0.99) -> float:
    _eff, _pur, thr = aligned_eff_purity_thresholds(metrics)
    idx = threshold_index_for_efficiency(metrics, target_eff=target_eff)
    return float(thr[idx])


def evaluate_threshold(hits, targets, threshold: float):
    y_true = targets["hit_on_valid_particle"].to_numpy(dtype=bool)
    y_pred = hits["score_sigmoid"].to_numpy() >= threshold
    tp = np.sum(y_pred & y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    efficiency = tp / (tp + fn) if (tp + fn) else 0.0
    purity = tp / (tp + fp) if (tp + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    hit_retention = np.mean(y_pred)
    return {
        "threshold": float(threshold),
        "efficiency": float(efficiency),
        "purity": float(purity),
        "false_positive_rate": float(false_positive_rate),
        "hit_retention": float(hit_retention),
    }


def sample_thresholds_above(metrics: dict, base_threshold: float, n_points: int):
    _eff, _pur, thr = aligned_eff_purity_thresholds(metrics)
    unique_thr = np.unique(thr)
    candidates = unique_thr[unique_thr >= (base_threshold - 1e-12)]
    if candidates.size == 0:
        return np.asarray([base_threshold], dtype=float)
    if n_points <= 0 or candidates.size <= n_points:
        return candidates
    idx = np.linspace(0, candidates.size - 1, n_points, dtype=int)
    return candidates[np.unique(idx)]


def print_threshold_scan_report(name: str, target_eff: float, eff_low: float, eff_high: float, threshold_at_target: float, rows: list[dict]):
    print(
        f"\n{name}: threshold scan near target reconstructable-track efficiency "
        f"{target_eff:.3f} in [{eff_low:.3f}, {eff_high:.3f}] ({threshold_at_target:.4f})"
    )
    print("threshold\ttrack_eff\ttrack_pur\ttrack_fpr\ttrack_ret\thit_eff\thit_pur\thit_fpr\thit_ret")
    for row in rows:
        print(
            f"{row['threshold']:.4f}\t{row['track_efficiency']:.5f}\t{row['track_purity']:.5f}\t"
            f"{row['track_false_positive_rate']:.5f}\t{row['track_retention']:.5f}\t"
            f"{row['efficiency']:.5f}\t{row['purity']:.5f}\t{row['false_positive_rate']:.5f}\t{row['hit_retention']:.5f}"
        )


def add_threshold_scan_overlay(ax, rows: list[dict], colour: str, annotate: bool):
    if not rows:
        return
    eff = [row["efficiency"] for row in rows]
    pur = [row["purity"] for row in rows]
    ax.scatter(eff, pur, marker="x", s=22, color=colour, alpha=0.9)
    if annotate:
        for row in rows:
            ax.annotate(f"{row['threshold']:.3f}", (row["efficiency"], row["purity"]), fontsize=6, color=colour)


def build_scan_thresholds(
    track_scores: np.ndarray,
    target_eff: float,
    n_points: int,
    eff_window: float,
    max_track_eff: float,
) -> tuple[np.ndarray, float, float]:
    if track_scores.size == 0:
        raise ValueError("track_scores is empty, cannot build threshold scan.")
    max_track_eff = float(np.clip(max_track_eff, 0.0, 1.0))
    if max_track_eff <= 0.0:
        raise ValueError("Maximum achievable track efficiency is 0.0; cannot build threshold scan.")

    target_eff = float(np.clip(target_eff, 0.0, max_track_eff))
    eff_window = max(0.0, float(eff_window))
    eff_low = max(0.0, target_eff - eff_window)
    eff_high = min(max_track_eff, target_eff + eff_window)

    if n_points <= 1 or np.isclose(eff_low, eff_high):
        eff_points = np.asarray([target_eff], dtype=float)
    else:
        # Use descending efficiency points so thresholds naturally increase.
        eff_points = np.linspace(eff_high, eff_low, n_points, dtype=float)

    # track_scores contains only valid particles that can reach reconstructable status.
    # Convert absolute efficiency (over all valid particles) to conditional efficiency
    # over this subset before mapping to score quantiles.
    cond_eff_points = np.clip(eff_points / max_track_eff, 0.0, 1.0)
    thresholds = np.quantile(track_scores, 1.0 - cond_eff_points)
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = thresholds[np.isfinite(thresholds)]
    if thresholds.size == 0:
        raise ValueError("No finite thresholds were produced from the requested efficiency window.")
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))
    return thresholds, eff_low, eff_high


def write_threshold_scan_csv(csv_path: pathlib.Path, reports: dict[str, list[dict]]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "threshold",
                "track_efficiency",
                "track_purity",
                "track_false_positive_rate",
                "track_retention",
                "efficiency",
                "purity",
                "false_positive_rate",
                "hit_retention",
            ],
        )
        writer.writeheader()
        for model_name, rows in reports.items():
            for row in rows:
                writer.writerow({"model": model_name, **row})
    print(f"Wrote threshold scan metrics to {csv_path}")


args = parse_args()


# ------------------------------------------------------------------------------
# Cell 3 (markdown)
# # Filter model evaluation


# ------------------------------------------------------------------------------
# Cell 4 (markdown)
# ## Plot parameters


# ------------------------------------------------------------------------------
# Cell 5 (code)
training_colours = {
    "600 MeV eta 4": "mediumvioletred",
    "600 MeV eta 2.5": "cornflowerblue",
    # "1 GeV": "mediumseagreen", # |eta| < 2.5
    "900 MeV eta 4": "mediumseagreen",  # |eta| < 4.0
}

qty_bins = {
    "pt": np.array([0.6, 0.9, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
    "vz": np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi", "vz": "v_z"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": "", "vz": "[mm]"}
out_dir = "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/plots/"


# ------------------------------------------------------------------------------
# Cell 6 (markdown)
# ## Retrieve filtering model configuration


# ------------------------------------------------------------------------------
# Cell 7 (code)
with pathlib.Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/filtering.yaml").open() as f:
    fconfig = yaml.safe_load(f)

filter_params = ["particle_min_pt", "particle_max_abs_eta"]

print("name: " + fconfig["name"])
for i in filter_params:
    print("> " + i + "\t: ", fconfig["data"][i])


filtering_fnames = {
    "900 MeV eta 4": "/share/rcif2/pduckett/hepattn-dq/HF-pix-900MeV-eta4_20260214-T194855/train_eval.h5",
}
key = next(iter(filtering_fnames.keys()))
filtering_configs = {k: fconfig.copy() for k in filtering_fnames}

filter_inputs = ["hits_" + filtering_configs[key]["data"]["inputs"]["hit"][i] for i in range(len(filtering_configs[key]["data"]["inputs"]["hit"]))]
print("> inputs: ", filter_inputs)


# ------------------------------------------------------------------------------
# Cell 8 (markdown)
# ## Load evaluation file


# ------------------------------------------------------------------------------
# Cell 9 (code)
from plot_utils import binned, profile_plot


# ------------------------------------------------------------------------------
# Cell 10 (code)
def _sorted_event_keys(file_handle: h5py.File) -> list[str]:
    def _key(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    return sorted(file_handle.keys(), key=_key)


def compute_threshold_and_retention_from_h5(
    eval_path: str | pathlib.Path,
    model_threshold: float,
    target_eff: float = 0.99,
    max_events: int | None = None,
    reconstructable_min_hits: int = 3,
) -> tuple[float, float, np.ndarray, float]:
    """Compute reconstructable-track target-efficiency threshold and retained-hit fraction in one streaming pass."""
    target_eff = float(np.clip(target_eff, 0.0, 1.0))
    valid_particle_kth_scores = []
    n_total = 0
    n_keep_model = 0
    n_nonfinite_scores = 0

    with h5py.File(eval_path, "r") as f:
        keys = _sorted_event_keys(f)
        if max_events is not None:
            keys = keys[:max_events]

        for key in tqdm(keys, desc=f"Scanning events ({pathlib.Path(eval_path).name})"):
            score = np.asarray(f[f"{key}/preds/final/hit_filter/hit_on_valid_particle_prob"])[0].astype(np.float64, copy=False)
            is_true = np.asarray(f[f"{key}/targets/hit_on_valid_particle"])[0].astype(bool)
            particle_hit_valid = np.asarray(f[f"{key}/targets/particle_hit_valid"])[0].astype(bool)
            particle_valid = np.asarray(f[f"{key}/targets/particle_valid"])[0].astype(bool)
            finite = np.isfinite(score)
            n_nonfinite_scores += int(score.size - np.sum(finite))
            if np.any(finite):
                score_valid = score[finite]
                is_true_valid = is_true[finite]
                particle_hit_valid = particle_hit_valid[:, finite]
                n_total += score_valid.size
                n_keep_model += int(np.sum(score_valid >= model_threshold))

                # For each valid particle, summarize by the k-th highest kept-hit score.
                # If it has fewer than k associated hits, it can never be reconstructed.
                for particle_id in np.where(particle_valid)[0]:
                    particle_scores = score_valid[particle_hit_valid[particle_id]]
                    if particle_scores.size >= reconstructable_min_hits:
                        kth_score = np.partition(particle_scores, -reconstructable_min_hits)[-reconstructable_min_hits]
                        valid_particle_kth_scores.append(float(kth_score))
                    else:
                        valid_particle_kth_scores.append(float("-inf"))

    if not valid_particle_kth_scores:
        raise RuntimeError(f"No valid particles found in: {eval_path}")

    all_valid_kth_scores = np.asarray(valid_particle_kth_scores, dtype=np.float64)
    finite_track_scores = all_valid_kth_scores[np.isfinite(all_valid_kth_scores)]
    max_track_eff = float(finite_track_scores.size / all_valid_kth_scores.size)
    if finite_track_scores.size == 0:
        raise RuntimeError(f"No valid particles with >= {reconstructable_min_hits} associated hits found in: {eval_path}")

    if target_eff > max_track_eff:
        threshold_at_target_eff = 0.0
        print(
            f"Warning: requested target track efficiency {target_eff:.4f} exceeds maximum achievable "
            f"{max_track_eff:.4f}; using threshold 0.0."
        )
    else:
        conditional_target = target_eff / max_track_eff
        threshold_at_target_eff = float(np.quantile(finite_track_scores, 1.0 - conditional_target))

    frac_remaining_at_model_threshold = (n_keep_model / n_total) if n_total > 0 else 0.0
    if n_nonfinite_scores:
        print(f"Warning: ignored {n_nonfinite_scores} non-finite scores while computing threshold metrics.")
    return threshold_at_target_eff, frac_remaining_at_model_threshold, finite_track_scores, max_track_eff


def metrics_for_thresholds_from_h5(
    eval_path: str | pathlib.Path,
    thresholds: np.ndarray,
    max_events: int | None = None,
    reconstructable_min_hits: int = 3,
) -> list[dict]:
    """Compute hit-level and reconstructable-track-level metrics for multiple thresholds in one pass."""
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = thresholds[np.isfinite(thresholds)]
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))
    if thresholds.size == 0:
        raise ValueError("No finite thresholds provided for threshold-scan metrics.")

    tp = np.zeros_like(thresholds, dtype=np.int64)
    fp = np.zeros_like(thresholds, dtype=np.int64)
    fn = np.zeros_like(thresholds, dtype=np.int64)
    tn = np.zeros_like(thresholds, dtype=np.int64)
    track_tp = np.zeros_like(thresholds, dtype=np.int64)
    track_fp = np.zeros_like(thresholds, dtype=np.int64)
    track_fn = np.zeros_like(thresholds, dtype=np.int64)
    track_tn = np.zeros_like(thresholds, dtype=np.int64)
    n_nonfinite_scores = 0

    with h5py.File(eval_path, "r") as f:
        keys = _sorted_event_keys(f)
        if max_events is not None:
            keys = keys[:max_events]

        for key in tqdm(keys, desc=f"Scanning thresholds ({pathlib.Path(eval_path).name})"):
            score = np.asarray(f[f"{key}/preds/final/hit_filter/hit_on_valid_particle_prob"])[0].astype(np.float64, copy=False)
            y_true = np.asarray(f[f"{key}/targets/hit_on_valid_particle"])[0].astype(bool)
            particle_hit_valid = np.asarray(f[f"{key}/targets/particle_hit_valid"])[0].astype(bool)
            particle_valid = np.asarray(f[f"{key}/targets/particle_valid"])[0].astype(bool)
            finite = np.isfinite(score)
            n_nonfinite_scores += int(score.size - np.sum(finite))
            if not np.any(finite):
                continue
            score = score[finite]
            y_true = y_true[finite]
            particle_hit_valid = particle_hit_valid[:, finite]
            y_pred = score[:, None] >= thresholds[None, :]
            y_true_col = y_true[:, None]
            tp += np.sum(y_pred & y_true_col, axis=0)
            fp += np.sum(y_pred & ~y_true_col, axis=0)
            fn += np.sum(~y_pred & y_true_col, axis=0)
            tn += np.sum(~y_pred & ~y_true_col, axis=0)

            true_valid_particle = particle_valid
            pred_hits_per_particle = particle_hit_valid.astype(np.int32) @ y_pred.astype(np.int32)
            pred_reconstructable = pred_hits_per_particle >= reconstructable_min_hits
            true_valid_col = true_valid_particle[:, None]
            track_tp += np.sum(pred_reconstructable & true_valid_col, axis=0)
            track_fp += np.sum(pred_reconstructable & ~true_valid_col, axis=0)
            track_fn += np.sum(~pred_reconstructable & true_valid_col, axis=0)
            track_tn += np.sum(~pred_reconstructable & ~true_valid_col, axis=0)

    rows = []
    for i, threshold in enumerate(thresholds):
        recall_denom = tp[i] + fn[i]
        precision_denom = tp[i] + fp[i]
        fpr_denom = fp[i] + tn[i]
        total = tp[i] + fp[i] + fn[i] + tn[i]
        track_recall_denom = track_tp[i] + track_fn[i]
        track_precision_denom = track_tp[i] + track_fp[i]
        track_fpr_denom = track_fp[i] + track_tn[i]
        track_total = track_tp[i] + track_fp[i] + track_fn[i] + track_tn[i]
        rows.append(
            {
                "threshold": float(threshold),
                "track_efficiency": float(track_tp[i] / track_recall_denom) if track_recall_denom else 0.0,
                "track_purity": float(track_tp[i] / track_precision_denom) if track_precision_denom else 0.0,
                "track_false_positive_rate": float(track_fp[i] / track_fpr_denom) if track_fpr_denom else 0.0,
                "track_retention": float((track_tp[i] + track_fp[i]) / track_total) if track_total else 0.0,
                "efficiency": float(tp[i] / recall_denom) if recall_denom else 0.0,
                "purity": float(tp[i] / precision_denom) if precision_denom else 0.0,
                "false_positive_rate": float(fp[i] / fpr_denom) if fpr_denom else 0.0,
                "hit_retention": float((tp[i] + fp[i]) / total) if total else 0.0,
            }
        )
    if n_nonfinite_scores:
        print(f"Warning: ignored {n_nonfinite_scores} non-finite scores while computing threshold-scan metrics.")
    return rows


# Fast mode for threshold studies (avoids huge DataFrame concatenations).
RUN_FULL_PLOTS = args.run_full_plots
MAX_EVENTS = args.max_events
TARGET_EFF = args.target_efficiency
threshold = args.threshold

threshold_scan_reports = {}
for name, fname in filtering_fnames.items():
    model_threshold = filtering_configs[name]["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["threshold"]
    thr_eff, frac_remaining, track_scores, max_track_eff = compute_threshold_and_retention_from_h5(
        fname,
        model_threshold=model_threshold,
        target_eff=TARGET_EFF,
        max_events=MAX_EVENTS,
        reconstructable_min_hits=args.reconstructable_min_hits,
    )
    print(
        f"{name}: threshold for {TARGET_EFF * 100:.1f}% reconstructable-track efficiency "
        f"(>= {args.reconstructable_min_hits} hits) = {thr_eff:.4f}"
    )
    print(f"{name}: maximum achievable reconstructable-track efficiency = {max_track_eff:.4f}")
    print(f"{name}: fraction of hits remaining at model threshold {model_threshold:.4f} = {frac_remaining:.4f}")

    if args.report_threshold_scan:
        scan_thresholds, eff_low, eff_high = build_scan_thresholds(
            track_scores=track_scores,
            target_eff=TARGET_EFF,
            n_points=args.threshold_scan_points,
            eff_window=args.threshold_scan_eff_window,
            max_track_eff=max_track_eff,
        )
        rows = metrics_for_thresholds_from_h5(
            fname,
            thresholds=scan_thresholds,
            max_events=MAX_EVENTS,
            reconstructable_min_hits=args.reconstructable_min_hits,
        )
        threshold_scan_reports[name] = rows
        print_threshold_scan_report(name, min(TARGET_EFF, max_track_eff), eff_low, eff_high, thr_eff, rows)
    else:
        threshold_scan_reports[name] = []

if args.report_threshold_scan and args.threshold_scan_csv:
    write_threshold_scan_csv(pathlib.Path(args.threshold_scan_csv), threshold_scan_reports)

if (not RUN_FULL_PLOTS) or args.only_threshold_metrics:
    raise SystemExit(0)


from hit_evaluate import load_events

pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

filtering_results = {}
num_events = MAX_EVENTS
for name, fname in filtering_fnames.items():
    filtering_results[name] = load_events(fname=fname, randomize=num_events, write_inputs=None, write_parts=True, threshold=threshold)
    print("loaded")




# ------------------------------------------------------------------------------
# Cell 11 (markdown)
# ## Plotting metrics


# ------------------------------------------------------------------------------
# Cell 12 (markdown)
# ### Discriminant


# ------------------------------------------------------------------------------
# Cell 13 (code)
for name, (hits, targets, _parts, _metrics) in filtering_results.items():
    filter_threshold = filtering_configs[name]["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["threshold"]
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.hist(hits["score_sigmoid"][targets["hit_on_valid_particle"]], range=[0, 1], bins=40, density=True, color="C0", alpha=0.5, label="Valid hits")
    ax.hist(
        hits["score_sigmoid"][~targets["hit_on_valid_particle"]], range=[0, 1], bins=40, density=True, color="C1", alpha=0.5, label="Invalid hits"
    )

    ax.axvline(filter_threshold, color="r", ls="dashed", label=f"{filtering_configs[name]['name']} Threshold: {filter_threshold:.1f}")
    ax.set_xlabel("Discriminant score")
    ax.set_ylabel("Normalized counts")
    ax.set_xlim(-0.025, 1.025)
    ax.grid(which="both")
    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend()


# ------------------------------------------------------------------------------
# Cell 14 (markdown)
# ### Receiver operating characteristic


# ------------------------------------------------------------------------------
# Cell 15 (code)
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
for name, (_hits, _targets, _parts, metrics) in filtering_results.items():
    ax.plot(
        metrics["roc_fpr"],
        metrics["roc_tpr"],
        color=training_colours[name],
        label=f"{filtering_configs[name]['name']} {name}\nAUC: {metrics['roc_fpr_tpr_auc']:.4f}",
    )

    thid = np.argmin(np.abs(metrics["roc_fpr_tpr_thr"] - threshold))
    ax.scatter(metrics["roc_fpr"][thid], metrics["roc_tpr"][thid], color=training_colours[name], s=100)

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(-0.05, 1.01)
ax.set_ylim(0.0, 1.05)
ax.grid(which="both")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend()


# ------------------------------------------------------------------------------
# Cell 16 (markdown)
# ### Efficiency purity plot


# ------------------------------------------------------------------------------
# Cell 17 (code)
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
for name, (_hits, _targets, _parts, metrics) in filtering_results.items():
    curve_eff, curve_pur, _curve_thr = hit_purity_efficiency_curve(metrics)
    ax.plot(
        curve_eff,
        curve_pur,
        color=training_colours[name],
        label=f"{filtering_configs[name]['name']} {name}\nAUC: {metrics['roc_eff_pur_auc']:.4f}",
    )

    marker_eff, marker_pur, _marker_thr = hit_curve_point_for_threshold(metrics, threshold)
    ax.scatter(marker_eff, marker_pur, color=training_colours[name], s=100)
    add_threshold_scan_overlay(
        ax,
        threshold_scan_reports.get(name, []),
        colour=training_colours[name],
        annotate=args.annotate_threshold_scan,
    )

ax.set_xlabel("Hit Efficiency")
ax.set_ylabel("Hit Purity")
ax.set_xlim(0.9, 1.01)
ax.set_ylim(0.3, 1.01)
ax.grid(which="both")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend()
fig.savefig(pathlib.Path(out_dir) / "filter_hit_purity_efficiency.pdf")


# ------------------------------------------------------------------------------
# Cell 18 (markdown)
# ### Particle efficiency (pT binned)


# ------------------------------------------------------------------------------
# Cell 19 (code)
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

for name, (_hits, _targets, parts, _metrics) in filtering_results.items():
    reconstructable = np.where(parts["pred_hits"] >= 3, True, False)  # reconstructable particles must have >=3 hits
    reconstructable = reconstructable & parts["valid"]  # apply valid_particle selection
    valid = ~np.isnan(parts["particle_pt"])  # remove excess entries (particles in event less than n_max_particles)
    bin_count, bin_error = binned(reconstructable[valid], parts["particle_pt"][valid], qty_bins["pt"])
    profile_plot(bin_count, bin_error, qty_bins["pt"], axes=ax, colour=training_colours[name], label=f"{filtering_configs[name]['name']} {name}")

ax.set_xlabel(rf"Particle ${qty_symbols['pt']}$ {qty_units['pt']}")
ax.set_ylabel("Reconstructable particles")
ax.set_ylim(0.97, 1)
ax.set_xticks(np.arange(start=2, stop=11, step=2))
ax.grid(which="both")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend(loc=3)
plt.show()


# ------------------------------------------------------------------------------
# Cell 20 (markdown)
# ### Combined plot


# ------------------------------------------------------------------------------
# Cell 21 (code)


# ------------------------------------------------------------------------------
# Cell 22 (code)
fig, ax = plt.subplots(ncols=2, figsize=(10, 3), constrained_layout=True)
for name, (_hits, _targets, parts, metrics) in filtering_results.items():
    curve_eff, curve_pur, _curve_thr = hit_purity_efficiency_curve(metrics)
    ax[0].plot(
        curve_eff,
        curve_pur,
        color=training_colours[name],
        label=f"{filtering_configs[name]['name']} {name}\nAUC: {metrics['roc_eff_pur_auc']:.4f}",
    )
    marker_eff, marker_pur, _marker_thr = hit_curve_point_for_threshold(metrics, threshold)
    ax[0].scatter(marker_eff, marker_pur, color=training_colours[name], s=100)
    add_threshold_scan_overlay(
        ax[0],
        threshold_scan_reports.get(name, []),
        colour=training_colours[name],
        annotate=args.annotate_threshold_scan,
    )

    # reconstructable particles must have >=3 hits
    reconstructable = np.where(parts["pred_hits"] >= 3, True, False)
    # apply valid_particle selection
    reconstructable = reconstructable & parts["valid"]
    # remove excess entries (particles in event less than n_max_particles)
    valid = ~np.isnan(parts["particle_pt"])
    bin_count, bin_error = binned(reconstructable[valid], parts["particle_pt"][valid], qty_bins["pt"])
    profile_plot(bin_count, bin_error, qty_bins["pt"], axes=ax[1], colour=training_colours[name], label=f"{filtering_configs[name]['name']} {name}")

ax[0].set_xlabel("Hit Efficiency")
ax[0].set_ylabel("Hit Purity")
ax[0].set_xlim(0.96, 1.0)
ax[0].set_ylim(0.5, 1.01)
ax[0].grid(which="both")
ax[0].grid(zorder=0, alpha=0.25, linestyle="--")
ax[0].legend(loc=3)

ax[1].set_xlabel(rf"Particle ${qty_symbols['pt']}$ {qty_units['pt']}")
ax[1].set_ylabel("Reconstructable Particles")
ax[1].set_ylim(0.97, 1)
ax[1].set_xticks(np.arange(start=2, stop=11, step=2))
ax[1].grid(which="both")
ax[1].grid(zorder=0, alpha=0.25, linestyle="--")
ax[1].legend(loc=3)

fig.savefig(pathlib.Path(out_dir) / "filter_response.pdf")
plt.show()


# ------------------------------------------------------------------------------
# Cell 23 (code)
# calculate the threshold that gives target hit efficiency
for name, (_hits, _targets, _parts, metrics) in filtering_results.items():
    target_threshold = threshold_for_efficiency(metrics, target_eff=TARGET_EFF)
    print(f"{name} threshold for {TARGET_EFF * 100:.1f}% hit efficiency: {target_threshold:.4f}")


# ------------------------------------------------------------------------------
# Cell 24 (code)
# calculate fraction of hits remaining after filtering
for name, (hits, _targets, _parts, _metrics) in filtering_results.items():
    filter_threshold = filtering_configs[name]["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["threshold"]
    num_hits_before = hits.shape[0]
    num_hits_after = np.sum(hits["score_sigmoid"] >= filter_threshold)
    fraction_remaining = num_hits_after / num_hits_before
    print(f"{name} fraction of hits remaining after filtering: {fraction_remaining:.4f}")


# ------------------------------------------------------------------------------
# Cell 25 (code)
