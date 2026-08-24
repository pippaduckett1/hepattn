"""Compare first-hit prediction performance across TrackML test eval HDF5 runs.

Expected input files:
  run_firsthitpred.py test --config firsthitpred.yaml --ckpt_path <...>

Per run, this script creates:
1) TP/FN/FP counts vs eta
2) first-hit recall/precision vs eta
3) eta-r heatmaps for TP/FN/FP
4) score distributions and threshold curves (if hit_is_first_prob exists)
5) top-k budget metrics/plots from hit_is_first_prob (if configured)
6) summary.csv with aggregate metrics

Across runs, this script creates:
1) global metric comparison bars (recall, precision, f1, fpr)
2) recall vs eta comparison
3) precision vs eta comparison
4) precision-recall curve comparison (if probabilities exist)
5) top-k comparison curves across runs (if configured)
6) summary_all_runs.csv and summary_all_runs_topk.csv (if configured)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg", force=True)


@dataclass
class RunConfig:
    name: str
    eval_h5: Path
    color: str | None = None


@dataclass
class PlotConfig:
    runs: list[RunConfig]
    out_dir: Path | None = None
    num_events: int = -1
    eta_bins: int = 80
    r_bins: int = 80
    eta_range: tuple[float, float] = (-5.0, 5.0)
    r_range: tuple[float, float] = (0.0, 1.5)
    score_bins: int = 200
    topk_limits: list[int] | None = None
    topk_policy: str = "global_topk"
    topk_eta_bins: int = 10
    topk_r_bins: int = 8
    topk_eta_range: tuple[float, float] = (-5.0, 5.0)
    topk_r_range: tuple[float, float] = (0.0, 1.5)
    topk_min_prob: float | None = None
    topk_fallback_when_empty: bool = True
    write_per_run_plots: bool = True


@dataclass
class RunResult:
    name: str
    color: str | None
    events_processed: int
    tp_total: float
    fn_total: float
    fp_total: float
    tn_total: float
    h_eta_tp: np.ndarray
    h_eta_fn: np.ndarray
    h_eta_fp: np.ndarray
    h_eta_true: np.ndarray
    h_eta_pred: np.ndarray
    h2_tp: np.ndarray
    h2_fn: np.ndarray
    h2_fp: np.ndarray
    h_score_pos: np.ndarray
    h_score_neg: np.ndarray
    topk_limits: np.ndarray
    topk_policy_tp: np.ndarray
    topk_policy_fp: np.ndarray
    topk_policy_fn: np.ndarray
    topk_policy_selected: np.ndarray
    topk_global_tp: np.ndarray
    topk_global_fp: np.ndarray
    topk_global_fn: np.ndarray
    topk_global_selected: np.ndarray
    topk_true: np.ndarray
    saw_probs: bool


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return slug.strip("_") or "run"


def _has_dataset(group: h5py.Group, path: str) -> bool:
    try:
        _ = group[path]
    except KeyError:
        return False
    return True


def _read_array(group: h5py.Group, path: str, required: bool = True) -> np.ndarray | None:
    if not _has_dataset(group, path):
        if required:
            raise KeyError(f"Missing dataset '{path}' in event '{group.name}'")
        return None
    arr = np.asarray(group[path])
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _sorted_event_keys(h5f: h5py.File) -> list[str]:
    def _key(k: str) -> tuple[int, int | str]:
        return (0, int(k)) if k.isdigit() else (1, k)

    return sorted(h5f.keys(), key=_key)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _metrics_from_confusion(tp: float, fn: float, fp: float, tn: float) -> dict[str, float]:
    pos_total = tp + fn
    pred_pos_total = tp + fp
    neg_total = fp + tn
    total = tp + fn + fp + tn

    recall = tp / pos_total if pos_total > 0 else 0.0
    precision = tp / pred_pos_total if pred_pos_total > 0 else 0.0
    specificity = tn / neg_total if neg_total > 0 else 0.0
    fpr = fp / neg_total if neg_total > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "accuracy": accuracy,
    }


def _build_global_topk_mask(
    prob: np.ndarray,
    k: int,
    min_prob: float | None = None,
    fallback_when_empty: bool = True,
) -> np.ndarray:
    selected = np.zeros(prob.shape[0], dtype=bool)
    if prob.size == 0 or k <= 0:
        return selected

    candidates = np.where(prob >= float(min_prob))[0] if min_prob is not None else np.arange(prob.size)
    if candidates.size == 0:
        if not fallback_when_empty:
            return selected
        candidates = np.arange(prob.size)

    k_eff = min(int(k), int(candidates.size))
    if k_eff == candidates.size:
        selected[candidates] = True
        return selected
    local = prob[candidates]
    idx_local = np.argpartition(local, local.size - k_eff)[local.size - k_eff :]
    selected[candidates[idx_local]] = True
    return selected


def _build_eta_r_topk_mask(
    prob: np.ndarray,
    eta: np.ndarray,
    r: np.ndarray,
    k: int,
    settings: PlotConfig,
    min_prob: float | None = None,
    fallback_when_empty: bool = True,
) -> np.ndarray:
    selected = np.zeros(prob.shape[0], dtype=bool)
    if prob.size == 0 or k <= 0:
        return selected

    candidate_idx = np.where(prob >= float(min_prob))[0] if min_prob is not None else np.arange(prob.size)
    if candidate_idx.size == 0:
        if not fallback_when_empty:
            return selected
        candidate_idx = np.arange(prob.size)

    prob_c = prob[candidate_idx]
    eta_c = eta[candidate_idx]
    r_c = r[candidate_idx]

    k_eff = min(int(k), int(prob_c.size))
    if k_eff == prob_c.size:
        selected[candidate_idx] = True
        return selected

    eta_edges = np.linspace(settings.topk_eta_range[0], settings.topk_eta_range[1], settings.topk_eta_bins + 1)
    r_edges = np.linspace(settings.topk_r_range[0], settings.topk_r_range[1], settings.topk_r_bins + 1)

    eta_bin = np.digitize(eta_c, eta_edges, right=False) - 1
    r_bin = np.digitize(r_c, r_edges, right=False) - 1
    eta_bin = np.clip(eta_bin, 0, settings.topk_eta_bins - 1)
    r_bin = np.clip(r_bin, 0, settings.topk_r_bins - 1)

    num_bins = settings.topk_eta_bins * settings.topk_r_bins
    bin_id = eta_bin * settings.topk_r_bins + r_bin
    bin_counts = np.bincount(bin_id, minlength=num_bins).astype(np.int64)
    total_candidates = int(bin_counts.sum())
    if total_candidates == 0:
        return _build_global_topk_mask(prob, k_eff, min_prob=min_prob, fallback_when_empty=fallback_when_empty)

    quota_float = bin_counts.astype(np.float64) * (k_eff / float(total_candidates))
    bin_quota = np.floor(quota_float).astype(np.int64)
    bin_quota = np.minimum(bin_quota, bin_counts)

    remaining_slots = int(k_eff - int(bin_quota.sum()))
    if remaining_slots > 0:
        frac = quota_float - bin_quota.astype(np.float64)
        has_capacity = bin_counts > bin_quota
        frac = np.where(has_capacity, frac, -1.0)
        order = np.argsort(frac)[::-1]
        for idx in order:
            if remaining_slots == 0:
                break
            if frac[idx] < 0:
                break
            if bin_counts[idx] > bin_quota[idx]:
                bin_quota[idx] += 1
                remaining_slots -= 1

    for current_bin in np.where(bin_quota > 0)[0]:
        local = np.where(bin_id == current_bin)[0]
        if local.size == 0:
            continue
        take_k = min(int(bin_quota[current_bin]), int(local.size))
        if take_k == local.size:
            selected[candidate_idx[local]] = True
        elif take_k > 0:
            local_top = np.argpartition(prob_c[local], local.size - take_k)[local.size - take_k :]
            selected[candidate_idx[local[local_top]]] = True

    selected_n = int(selected.sum())
    if selected_n < k_eff:
        remaining = np.where(~selected)[0]
        if remaining.size > 0:
            need = min(k_eff - selected_n, remaining.size)
            top_local = np.argpartition(prob[remaining], remaining.size - need)[remaining.size - need :]
            selected[remaining[top_local]] = True

    return selected


def _compute_topk_counts(selected: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float, float]:
    tp = float(np.sum(selected & truth))
    fp = float(np.sum(selected & (~truth)))
    fn = float(np.sum((~selected) & truth))
    selected_total = float(np.sum(selected))
    true_total = float(np.sum(truth))
    return tp, fp, fn, selected_total, true_total


def _topk_scores(tp: np.ndarray, selected: np.ndarray, true_total: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recall = _safe_divide(tp, true_total)
    precision = _safe_divide(tp, selected)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return recall, precision, f1


def _analyze_run(run: RunConfig, settings: PlotConfig, eta_edges: np.ndarray, r_edges: np.ndarray, score_edges: np.ndarray) -> RunResult:
    if not run.eval_h5.exists():
        raise FileNotFoundError(f"Eval file does not exist for run '{run.name}': {run.eval_h5}")

    h_eta_tp = np.zeros(settings.eta_bins, dtype=np.float64)
    h_eta_fn = np.zeros(settings.eta_bins, dtype=np.float64)
    h_eta_fp = np.zeros(settings.eta_bins, dtype=np.float64)
    h_eta_true = np.zeros(settings.eta_bins, dtype=np.float64)
    h_eta_pred = np.zeros(settings.eta_bins, dtype=np.float64)

    h2_tp = np.zeros((settings.eta_bins, settings.r_bins), dtype=np.float64)
    h2_fn = np.zeros((settings.eta_bins, settings.r_bins), dtype=np.float64)
    h2_fp = np.zeros((settings.eta_bins, settings.r_bins), dtype=np.float64)

    h_score_pos = np.zeros(settings.score_bins, dtype=np.float64)
    h_score_neg = np.zeros(settings.score_bins, dtype=np.float64)

    topk_limits = np.asarray(sorted({int(k) for k in (settings.topk_limits or []) if int(k) > 0}), dtype=np.int64)
    n_k = int(topk_limits.size)
    topk_policy_tp = np.zeros(n_k, dtype=np.float64)
    topk_policy_fp = np.zeros(n_k, dtype=np.float64)
    topk_policy_fn = np.zeros(n_k, dtype=np.float64)
    topk_policy_selected = np.zeros(n_k, dtype=np.float64)
    topk_global_tp = np.zeros(n_k, dtype=np.float64)
    topk_global_fp = np.zeros(n_k, dtype=np.float64)
    topk_global_fn = np.zeros(n_k, dtype=np.float64)
    topk_global_selected = np.zeros(n_k, dtype=np.float64)
    topk_true = np.zeros(n_k, dtype=np.float64)

    saw_probs = False
    tp_total = 0.0
    fn_total = 0.0
    fp_total = 0.0
    tn_total = 0.0

    with h5py.File(run.eval_h5, "r") as h5f:
        event_keys = _sorted_event_keys(h5f)
        if settings.num_events > 0:
            event_keys = event_keys[: settings.num_events]
        if len(event_keys) == 0:
            raise RuntimeError(f"No events found in eval file: {run.eval_h5}")

        for i, key in enumerate(event_keys):
            g = h5f[key]
            pred = _read_array(g, "preds/final/query_init/hit_is_first").astype(bool)
            truth = _read_array(g, "targets/hit_is_first").astype(bool)
            eta = _read_array(g, "inputs/hit_eta").astype(np.float64)
            r = _read_array(g, "inputs/hit_r").astype(np.float64)
            valid = _read_array(g, "targets/hit_valid", required=False)
            valid_mask = valid.astype(bool) if valid is not None else np.ones_like(truth, dtype=bool)

            if pred.shape != truth.shape or pred.shape != valid_mask.shape:
                raise ValueError(f"Shape mismatch in run '{run.name}', event '{key}': pred {pred.shape}, truth {truth.shape}, valid {valid_mask.shape}")

            pred = pred[valid_mask]
            truth = truth[valid_mask]
            eta = eta[valid_mask]
            r = r[valid_mask]

            tp = pred & truth
            fn = (~pred) & truth
            fp = pred & (~truth)
            tn = (~pred) & (~truth)

            tp_total += float(tp.sum())
            fn_total += float(fn.sum())
            fp_total += float(fp.sum())
            tn_total += float(tn.sum())

            h_eta_tp += np.histogram(eta[tp], bins=eta_edges)[0]
            h_eta_fn += np.histogram(eta[fn], bins=eta_edges)[0]
            h_eta_fp += np.histogram(eta[fp], bins=eta_edges)[0]
            h_eta_true += np.histogram(eta[truth], bins=eta_edges)[0]
            h_eta_pred += np.histogram(eta[pred], bins=eta_edges)[0]

            h2_tp += np.histogram2d(eta[tp], r[tp], bins=[eta_edges, r_edges])[0]
            h2_fn += np.histogram2d(eta[fn], r[fn], bins=[eta_edges, r_edges])[0]
            h2_fp += np.histogram2d(eta[fp], r[fp], bins=[eta_edges, r_edges])[0]

            prob = _read_array(g, "preds/final/query_init/hit_is_first_prob", required=False)
            if prob is not None:
                prob = prob[valid_mask].astype(np.float64)
                h_score_pos += np.histogram(prob[truth], bins=score_edges)[0]
                h_score_neg += np.histogram(prob[~truth], bins=score_edges)[0]
                saw_probs = True

                if n_k > 0:
                    for ik, k in enumerate(topk_limits):
                        global_selected = _build_global_topk_mask(
                            prob,
                            int(k),
                            min_prob=settings.topk_min_prob,
                            fallback_when_empty=settings.topk_fallback_when_empty,
                        )
                        g_tp, g_fp, g_fn, g_sel, g_true = _compute_topk_counts(global_selected, truth)
                        topk_global_tp[ik] += g_tp
                        topk_global_fp[ik] += g_fp
                        topk_global_fn[ik] += g_fn
                        topk_global_selected[ik] += g_sel

                        if settings.topk_policy == "eta_r_topk":
                            policy_selected = _build_eta_r_topk_mask(
                                prob,
                                eta,
                                r,
                                int(k),
                                settings,
                                min_prob=settings.topk_min_prob,
                                fallback_when_empty=settings.topk_fallback_when_empty,
                            )
                        else:
                            policy_selected = global_selected

                        p_tp, p_fp, p_fn, p_sel, _ = _compute_topk_counts(policy_selected, truth)
                        topk_policy_tp[ik] += p_tp
                        topk_policy_fp[ik] += p_fp
                        topk_policy_fn[ik] += p_fn
                        topk_policy_selected[ik] += p_sel
                        topk_true[ik] += g_true

            if (i + 1) % 50 == 0 or (i + 1) == len(event_keys):
                print(f"[{run.name}] Processed {i + 1}/{len(event_keys)} events")

    return RunResult(
        name=run.name,
        color=run.color,
        events_processed=len(event_keys),
        tp_total=tp_total,
        fn_total=fn_total,
        fp_total=fp_total,
        tn_total=tn_total,
        h_eta_tp=h_eta_tp,
        h_eta_fn=h_eta_fn,
        h_eta_fp=h_eta_fp,
        h_eta_true=h_eta_true,
        h_eta_pred=h_eta_pred,
        h2_tp=h2_tp,
        h2_fn=h2_fn,
        h2_fp=h2_fp,
        h_score_pos=h_score_pos,
        h_score_neg=h_score_neg,
        topk_limits=topk_limits,
        topk_policy_tp=topk_policy_tp,
        topk_policy_fp=topk_policy_fp,
        topk_policy_fn=topk_policy_fn,
        topk_policy_selected=topk_policy_selected,
        topk_global_tp=topk_global_tp,
        topk_global_fp=topk_global_fp,
        topk_global_fn=topk_global_fn,
        topk_global_selected=topk_global_selected,
        topk_true=topk_true,
        saw_probs=saw_probs,
    )


def _save_eta_outcome_plot(out_dir: Path, run_name: str, eta_edges: np.ndarray, h_eta_tp: np.ndarray, h_eta_fn: np.ndarray, h_eta_fp: np.ndarray) -> None:
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(eta_centers, h_eta_tp, where="mid", label="TP (correct first-hit predictions)")
    ax.step(eta_centers, h_eta_fn, where="mid", label="FN (missed true first hits)")
    ax.step(eta_centers, h_eta_fp, where="mid", label="FP (incorrect first-hit predictions)")
    ax.set_xlabel("eta")
    ax.set_ylabel("count")
    ax.set_title(f"{run_name}: first-hit prediction outcomes vs eta")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_outcomes_eta.png", dpi=180)
    plt.close(fig)


def _save_eta_effpur_plot(out_dir: Path, run_name: str, eta_edges: np.ndarray, h_eta_tp: np.ndarray, h_eta_true: np.ndarray, h_eta_pred: np.ndarray) -> None:
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    eta_width = eta_edges[1] - eta_edges[0]
    eta_recall = _safe_divide(h_eta_tp, h_eta_true)
    eta_precision = _safe_divide(h_eta_tp, h_eta_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(eta_centers - 0.5 * eta_width, eta_recall, width=eta_width, label="recall", alpha=0.7)
    ax.bar(eta_centers + 0.5 * eta_width, eta_precision, width=eta_width, label="precision", alpha=0.7)
    ax.set_xlabel("eta")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{run_name}: first-hit recall/precision vs eta")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_recall_precision_eta.png", dpi=180)
    plt.close(fig)


def _save_eta_r_outcome_plot(
    out_dir: Path,
    run_name: str,
    eta_edges: np.ndarray,
    r_edges: np.ndarray,
    h2_tp: np.ndarray,
    h2_fn: np.ndarray,
    h2_fp: np.ndarray,
) -> None:
    extent = [eta_edges[0], eta_edges[-1], r_edges[0], r_edges[-1]]
    maps = [("TP (correct)", h2_tp, "Greens"), ("FN (missed)", h2_fn, "Blues"), ("FP (incorrect)", h2_fp, "Reds")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, (title, arr, cmap) in zip(axes, maps):
        im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap=cmap)
        ax.set_xlabel("eta")
        ax.set_ylabel("r [m]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"{run_name}: first-hit prediction outcomes in eta-r")
    fig.savefig(out_dir / "first_hit_outcomes_eta_r.png", dpi=180)
    plt.close(fig)


def _threshold_curves(score_edges: np.ndarray, h_score_pos: np.ndarray, h_score_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_total = h_score_pos.sum()
    tp = np.cumsum(h_score_pos[::-1])[::-1]
    fp = np.cumsum(h_score_neg[::-1])[::-1]
    precision = _safe_divide(tp, tp + fp)
    recall = tp / pos_total if pos_total > 0 else np.zeros_like(tp, dtype=float)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return recall, precision, f1


def _save_score_plots(
    out_dir: Path,
    run_name: str,
    score_edges: np.ndarray,
    h_score_pos: np.ndarray,
    h_score_neg: np.ndarray,
) -> tuple[float, float, float] | None:
    pos_total = h_score_pos.sum()
    neg_total = h_score_neg.sum()
    if (pos_total + neg_total) == 0:
        return None

    centers = 0.5 * (score_edges[:-1] + score_edges[1:])
    width = score_edges[1] - score_edges[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, h_score_neg, width=width, alpha=0.5, label="true non-first hits")
    ax.bar(centers, h_score_pos, width=width, alpha=0.5, label="true first hits")
    ax.set_xlabel("predicted P(is_first)")
    ax.set_ylabel("count")
    ax.set_title(f"{run_name}: first-hit score distribution")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_score_distribution.png", dpi=180)
    plt.close(fig)

    recall, precision, f1 = _threshold_curves(score_edges, h_score_pos, h_score_neg)
    thresholds = score_edges[:-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, f1, label="f1")
    ax.set_xlabel("threshold on P(is_first)")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{run_name}: threshold scan for first-hit prediction")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_threshold_scan.png", dpi=180)
    plt.close(fig)

    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def _save_topk_plots(out_dir: Path, run_name: str, result: RunResult) -> None:
    if result.topk_limits.size == 0:
        return

    p_rec, p_prec, _ = _topk_scores(result.topk_policy_tp, result.topk_policy_selected, result.topk_true)
    g_rec, g_prec, _ = _topk_scores(result.topk_global_tp, result.topk_global_selected, result.topk_true)

    k = result.topk_limits.astype(np.int64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k, p_rec, marker="o", label="policy recall")
    ax.plot(k, p_prec, marker="o", label="policy precision")
    ax.plot(k, g_rec, marker="o", linestyle="--", label="global recall")
    ax.plot(k, g_prec, marker="o", linestyle="--", label="global precision")
    ax.set_xlabel("top-k query budget")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{run_name}: top-k budget performance")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_topk_scores.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k, p_rec - g_rec, marker="o", label="recall delta (policy-global)")
    ax.plot(k, p_prec - g_prec, marker="o", label="precision delta (policy-global)")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("top-k query budget")
    ax.set_ylabel("delta")
    ax.set_title(f"{run_name}: top-k policy minus global baseline")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_hit_topk_policy_minus_global.png", dpi=180)
    plt.close(fig)


def _write_summary(output_path: Path, metrics: dict[str, float], best_threshold_metrics: tuple[float, float, float] | None, result: RunResult) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ["tp", "fn", "fp", "tn", "recall", "precision", "f1", "specificity", "fpr", "accuracy"]:
            writer.writerow([k, metrics[k]])
        if best_threshold_metrics is not None:
            best_thr, best_prec, best_rec = best_threshold_metrics
            writer.writerow(["best_threshold_by_f1", best_thr])
            writer.writerow(["best_threshold_precision", best_prec])
            writer.writerow(["best_threshold_recall", best_rec])

        if result.topk_limits.size > 0:
            p_rec, p_prec, _ = _topk_scores(result.topk_policy_tp, result.topk_policy_selected, result.topk_true)
            g_rec, g_prec, _ = _topk_scores(result.topk_global_tp, result.topk_global_selected, result.topk_true)
            writer.writerow([])
            writer.writerow([
                "topk",
                "policy_tp",
                "policy_fp",
                "policy_fn",
                "policy_recall",
                "policy_precision",
                "global_tp",
                "global_fp",
                "global_fn",
                "global_recall",
                "global_precision",
                "policy_minus_global_recall",
                "policy_minus_global_precision",
            ])
            for i, k in enumerate(result.topk_limits.tolist()):
                writer.writerow([
                    int(k),
                    float(result.topk_policy_tp[i]),
                    float(result.topk_policy_fp[i]),
                    float(result.topk_policy_fn[i]),
                    float(p_rec[i]),
                    float(p_prec[i]),
                    float(result.topk_global_tp[i]),
                    float(result.topk_global_fp[i]),
                    float(result.topk_global_fn[i]),
                    float(g_rec[i]),
                    float(g_prec[i]),
                    float(p_rec[i] - g_rec[i]),
                    float(p_prec[i] - g_prec[i]),
                ])


def _save_global_metric_comparison(out_dir: Path, results: list[RunResult]) -> None:
    metric_names = ["recall", "precision", "f1", "fpr"]
    x = np.arange(len(results))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric_name in enumerate(metric_names):
        values = []
        for r in results:
            m = _metrics_from_confusion(r.tp_total, r.fn_total, r.fp_total, r.tn_total)
            values.append(m[metric_name])
        ax.bar(x + (i - 1.5) * width, values, width=width, label=metric_name)

    ax.set_xticks(x)
    ax.set_xticklabels([r.name for r in results], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Comparison across runs: global first-hit metrics")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_global_metrics.png", dpi=180)
    plt.close(fig)


def _save_eta_curve_comparison(out_dir: Path, eta_edges: np.ndarray, results: list[RunResult], metric: str) -> None:
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        if metric == "recall":
            y = _safe_divide(r.h_eta_tp, r.h_eta_true)
        elif metric == "precision":
            y = _safe_divide(r.h_eta_tp, r.h_eta_pred)
        else:
            raise ValueError(f"Unknown eta metric: {metric}")
        ax.plot(eta_centers, y, label=r.name, color=r.color)

    ax.set_xlabel("eta")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Comparison across runs: first-hit {metric} vs eta")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_eta_{metric}.png", dpi=180)
    plt.close(fig)


def _save_pr_curve_comparison(out_dir: Path, score_edges: np.ndarray, results: list[RunResult]) -> None:
    valid_results = [r for r in results if r.saw_probs and (r.h_score_pos.sum() + r.h_score_neg.sum()) > 0]
    if len(valid_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in valid_results:
        recall, precision, _ = _threshold_curves(score_edges, r.h_score_pos, r.h_score_neg)
        ax.plot(recall, precision, label=r.name, color=r.color)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Comparison across runs: precision-recall curve")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_precision_recall_curves.png", dpi=180)
    plt.close(fig)


def _save_topk_curve_comparison(out_dir: Path, results: list[RunResult], metric: str, source: str) -> None:
    valid_results = [r for r in results if r.topk_limits.size > 0]
    if len(valid_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in valid_results:
        if source == "policy":
            tp = r.topk_policy_tp
            selected = r.topk_policy_selected
        elif source == "global":
            tp = r.topk_global_tp
            selected = r.topk_global_selected
        else:
            raise ValueError(f"Unknown topk source: {source}")

        recall, precision, _ = _topk_scores(tp, selected, r.topk_true)
        y = recall if metric == "recall" else precision
        ax.plot(r.topk_limits, y, marker="o", label=r.name, color=r.color)

    ax.set_xlabel("top-k query budget")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Comparison across runs: {source} top-k {metric}")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_topk_{source}_{metric}.png", dpi=180)
    plt.close(fig)


def _save_topk_delta_comparison(out_dir: Path, results: list[RunResult], metric: str) -> None:
    valid_results = [r for r in results if r.topk_limits.size > 0]
    if len(valid_results) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in valid_results:
        p_rec, p_prec, _ = _topk_scores(r.topk_policy_tp, r.topk_policy_selected, r.topk_true)
        g_rec, g_prec, _ = _topk_scores(r.topk_global_tp, r.topk_global_selected, r.topk_true)
        delta = (p_rec - g_rec) if metric == "recall" else (p_prec - g_prec)
        ax.plot(r.topk_limits, delta, marker="o", label=r.name, color=r.color)

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("top-k query budget")
    ax.set_ylabel("delta (policy-global)")
    ax.set_title(f"Comparison across runs: top-k {metric} delta")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_topk_policy_minus_global_{metric}.png", dpi=180)
    plt.close(fig)


def _write_all_runs_summary(out_path: Path, results: list[RunResult]) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "events", "tp", "fn", "fp", "tn", "recall", "precision", "f1", "specificity", "fpr", "accuracy"])
        for r in results:
            m = _metrics_from_confusion(r.tp_total, r.fn_total, r.fp_total, r.tn_total)
            writer.writerow([r.name, r.events_processed, m["tp"], m["fn"], m["fp"], m["tn"], m["recall"], m["precision"], m["f1"], m["specificity"], m["fpr"], m["accuracy"]])


def _write_all_runs_topk_summary(out_path: Path, results: list[RunResult]) -> None:
    valid_results = [r for r in results if r.topk_limits.size > 0]
    if len(valid_results) == 0:
        return

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "topk",
                "policy_tp",
                "policy_fp",
                "policy_fn",
                "policy_recall",
                "policy_precision",
                "global_tp",
                "global_fp",
                "global_fn",
                "global_recall",
                "global_precision",
                "policy_minus_global_recall",
                "policy_minus_global_precision",
            ]
        )
        for r in valid_results:
            p_rec, p_prec, _ = _topk_scores(r.topk_policy_tp, r.topk_policy_selected, r.topk_true)
            g_rec, g_prec, _ = _topk_scores(r.topk_global_tp, r.topk_global_selected, r.topk_true)
            for i, k in enumerate(r.topk_limits.tolist()):
                writer.writerow(
                    [
                        r.name,
                        int(k),
                        float(r.topk_policy_tp[i]),
                        float(r.topk_policy_fp[i]),
                        float(r.topk_policy_fn[i]),
                        float(p_rec[i]),
                        float(p_prec[i]),
                        float(r.topk_global_tp[i]),
                        float(r.topk_global_fp[i]),
                        float(r.topk_global_fn[i]),
                        float(g_rec[i]),
                        float(g_prec[i]),
                        float(p_rec[i] - g_rec[i]),
                        float(p_prec[i] - g_prec[i]),
                    ]
                )


def main() -> None:
    settings = PlotConfig(
        runs=[
            RunConfig(name="run_a", eval_h5=Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/first-hit-pred_20260227-T124450/ckpts/epoch=004-val_loss=0.04842_test_eval.h5"), color="tab:blue"),
            # RunConfig(name="run_b", eval_h5=Path("/path/to/run_b_test_eval.h5"), color="tab:orange"),
        ],
        out_dir=None,  # If None, use <first_run_parent>/first_hit_test_compare_plots
        num_events=-1,
        eta_bins=80,
        r_bins=80,
        eta_range=(-5.0, 5.0),
        r_range=(0.0, 1.5),
        score_bins=200,
        topk_limits=[600, 1200, 2200],
        topk_policy="eta_r_topk",  # one of: global_topk, eta_r_topk
        topk_eta_bins=10,
        topk_r_bins=8,
        topk_eta_range=(-5.0, 5.0),
        topk_r_range=(0.0, 1.5),
        topk_min_prob=0.05,  # require p(is_first) >= topk_min_prob before top-k
        topk_fallback_when_empty=True,  # model-like behavior: if none pass threshold, fallback to unconstrained top-k
        write_per_run_plots=True,
    )

    if len(settings.runs) == 0:
        raise ValueError("settings.runs is empty.")
    if settings.topk_policy not in {"global_topk", "eta_r_topk"}:
        raise ValueError(f"Unsupported topk_policy: {settings.topk_policy}")

    out_dir = settings.out_dir if settings.out_dir is not None else settings.runs[0].eval_h5.parent / "first_hit_test_compare_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    eta_edges = np.linspace(settings.eta_range[0], settings.eta_range[1], settings.eta_bins + 1)
    r_edges = np.linspace(settings.r_range[0], settings.r_range[1], settings.r_bins + 1)
    score_edges = np.linspace(0.0, 1.0, settings.score_bins + 1)

    results: list[RunResult] = []
    for run in settings.runs:
        result = _analyze_run(run, settings, eta_edges, r_edges, score_edges)
        results.append(result)

        if settings.write_per_run_plots:
            run_dir = out_dir / _slug(run.name)
            run_dir.mkdir(parents=True, exist_ok=True)

            _save_eta_outcome_plot(run_dir, run.name, eta_edges, result.h_eta_tp, result.h_eta_fn, result.h_eta_fp)
            _save_eta_effpur_plot(run_dir, run.name, eta_edges, result.h_eta_tp, result.h_eta_true, result.h_eta_pred)
            _save_eta_r_outcome_plot(run_dir, run.name, eta_edges, r_edges, result.h2_tp, result.h2_fn, result.h2_fp)

            best_threshold_metrics = None
            if result.saw_probs:
                best_threshold_metrics = _save_score_plots(run_dir, run.name, score_edges, result.h_score_pos, result.h_score_neg)

            if result.topk_limits.size > 0 and result.saw_probs:
                _save_topk_plots(run_dir, run.name, result)

            metrics = _metrics_from_confusion(result.tp_total, result.fn_total, result.fp_total, result.tn_total)
            _write_summary(run_dir / "summary.csv", metrics, best_threshold_metrics, result)

    _save_global_metric_comparison(out_dir, results)
    _save_eta_curve_comparison(out_dir, eta_edges, results, metric="recall")
    _save_eta_curve_comparison(out_dir, eta_edges, results, metric="precision")
    _save_pr_curve_comparison(out_dir, score_edges, results)
    _save_topk_curve_comparison(out_dir, results, metric="recall", source="policy")
    _save_topk_curve_comparison(out_dir, results, metric="precision", source="policy")
    _save_topk_curve_comparison(out_dir, results, metric="recall", source="global")
    _save_topk_curve_comparison(out_dir, results, metric="precision", source="global")
    _save_topk_delta_comparison(out_dir, results, metric="recall")
    _save_topk_delta_comparison(out_dir, results, metric="precision")
    _write_all_runs_summary(out_dir / "summary_all_runs.csv", results)
    _write_all_runs_topk_summary(out_dir / "summary_all_runs_topk.csv", results)

    print(f"Saved comparison plots to: {out_dir}")
    print(f"Saved all-runs summary to: {out_dir / 'summary_all_runs.csv'}")
    if any(r.topk_limits.size > 0 for r in results):
        print(f"Saved top-k summary to: {out_dir / 'summary_all_runs_topk.csv'}")


if __name__ == "__main__":
    main()
