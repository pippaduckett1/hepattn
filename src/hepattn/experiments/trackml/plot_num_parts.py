"""Plot per-particle hit and query count distributions from TrackML tracking eval HDF5.

This script is intended for outputs produced by:
  run_tracking.py test --config tracking.yaml --ckpt_path <...>

It creates:
1) distribution of number of truth hits per truth particle
2) distribution of number of queries per truth particle
3) profiles of each count vs |eta|
4) profiles of each count vs pT
5) eta-pT heatmaps of mean counts
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

mpl.use("Agg", force=True)

QuerySource = str
QUERY_AUTO = "auto"
QUERY_FINAL = "final"
QUERY_INIT_PRED = "query_init_pred"
QUERY_INIT_TRUTH = "query_init_truth"


@dataclass
class ParticleCounts:
    eta: np.ndarray
    pt: np.ndarray
    n_hits: np.ndarray
    n_queries: np.ndarray


@dataclass
class PlotConfig:
    eval_h5: Path
    config: Path | None = Path("src/hepattn/experiments/trackml/configs/tracking.yaml")
    out_dir: Path | None = None
    num_events: int = -1
    query_source: QuerySource = QUERY_AUTO
    first_hit_threshold: float | None = None
    eta_max: float | None = None
    pt_min: float | None = None
    pt_max: float | None = None
    n_eta_bins: int = 10
    n_pt_bins: int = 12


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


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


def _resolve_query_source(h5f: h5py.File, event_keys: list[str], query_source: QuerySource) -> QuerySource:
    if query_source != QUERY_AUTO:
        return query_source

    for key in event_keys:
        g = h5f[key]
        if _has_dataset(g, "preds/encoder/query_init/hit_is_first") or _has_dataset(g, "preds/encoder/query_init/hit_is_first_prob"):
            return QUERY_INIT_PRED
        if _has_dataset(g, "preds/final/track_valid/track_valid") and _has_dataset(g, "preds/final/track_hit_valid/track_hit_valid"):
            return QUERY_FINAL

    msg = "Could not infer query source from file (no encoder query_init preds and no final tracking preds found)."
    raise RuntimeError(msg)


def _map_hits_to_valid_particles(hit_particle_id: np.ndarray, valid_particle_ids: np.ndarray) -> np.ndarray:
    """Map per-hit particle IDs to compact [0, n_valid_particles) indices.

    Hits that are noise / invalid are assigned -1.
    """
    hit_particle_id = np.asarray(hit_particle_id)
    valid_particle_ids = np.asarray(valid_particle_ids)
    hit_owner = np.full(hit_particle_id.shape[0], -1, dtype=np.int32)

    if valid_particle_ids.size == 0:
        return hit_owner

    order = np.argsort(valid_particle_ids)
    sorted_ids = valid_particle_ids[order]
    pos = np.searchsorted(sorted_ids, hit_particle_id)
    in_range = pos < sorted_ids.size
    if not np.any(in_range):
        return hit_owner

    candidate_pos = pos[in_range]
    candidate_match = sorted_ids[candidate_pos] == hit_particle_id[in_range]
    if np.any(candidate_match):
        source_idx = np.flatnonzero(in_range)[candidate_match]
        compact_idx = order[candidate_pos[candidate_match]]
        hit_owner[source_idx] = compact_idx.astype(np.int32, copy=False)

    return hit_owner


def _extract_truth_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract per-particle arrays and per-hit owner index.

    Returns:
        valid_eta, valid_pt, n_hits_per_particle, hit_owner, n_particles
    """
    particle_valid = _read_array(group, "targets/particle_valid").astype(bool)
    particle_eta = _read_array(group, "targets/particle_eta").astype(np.float64)
    particle_pt = _read_array(group, "targets/particle_pt").astype(np.float64)

    valid_eta = particle_eta[particle_valid]
    valid_pt = particle_pt[particle_valid]
    n_particles = int(particle_valid.sum())

    particle_id = _read_array(group, "targets/particle_id", required=False)
    hit_particle_id = _read_array(group, "targets/hit_particle_id", required=False)

    if particle_id is not None and hit_particle_id is not None:
        valid_particle_ids = np.asarray(particle_id[particle_valid])
        hit_owner = _map_hits_to_valid_particles(hit_particle_id=np.asarray(hit_particle_id), valid_particle_ids=valid_particle_ids)
        owned = hit_owner[hit_owner >= 0]
        n_hits = np.bincount(owned, minlength=n_particles).astype(np.int32) if owned.size > 0 else np.zeros(n_particles, dtype=np.int32)
        return valid_eta, valid_pt, n_hits, hit_owner, n_particles

    # Fallback for older eval files that do not contain particle_id/hit_particle_id.
    particle_hit_valid = _read_array(group, "targets/particle_hit_valid").astype(bool)
    true_masks = particle_hit_valid[particle_valid]
    n_hits = true_masks.sum(axis=1).astype(np.int32)

    if n_particles == 0:
        hit_owner = np.full(particle_hit_valid.shape[-1], -1, dtype=np.int32)
    else:
        hit_owner = true_masks.argmax(axis=0).astype(np.int32, copy=False)
        hit_owner[~true_masks.any(axis=0)] = -1

    return valid_eta, valid_pt, n_hits, hit_owner, n_particles


def _counts_from_final_preds(group: h5py.Group, hit_owner: np.ndarray, n_particles: int) -> np.ndarray:
    counts = np.zeros(n_particles, dtype=np.int32)
    if n_particles == 0:
        return counts

    track_valid = _read_array(group, "preds/final/track_valid/track_valid").astype(bool)
    track_hit_valid = _read_array(group, "preds/final/track_hit_valid/track_hit_valid").astype(bool)

    query_mask = _read_array(group, "targets/query_mask", required=False)
    if query_mask is not None:
        track_valid = track_valid & query_mask.astype(bool)

    active_queries = np.flatnonzero(track_valid)
    if active_queries.size == 0:
        return counts

    num_hits = min(track_hit_valid.shape[1], hit_owner.shape[0])
    if num_hits == 0:
        return counts
    track_hit_valid = track_hit_valid[:, :num_hits]
    hit_owner = hit_owner[:num_hits]

    # Assign each active query to the truth particle with maximum overlap.
    # This avoids a dense [n_queries, n_particles] x n_hits matrix product.
    for query_idx in active_queries:
        owners = hit_owner[track_hit_valid[query_idx]]
        owners = owners[owners >= 0]
        if owners.size == 0:
            continue
        matched_particle = np.bincount(owners, minlength=n_particles).argmax()
        counts[matched_particle] += 1

    return counts


def _select_hits_for_query_init(
    group: h5py.Group,
    first_hit_threshold: float,
) -> np.ndarray:
    pred_first = _read_array(group, "preds/encoder/query_init/hit_is_first", required=False)
    pred_prob = _read_array(group, "preds/encoder/query_init/hit_is_first_prob", required=False)

    if pred_first is None and pred_prob is None:
        raise KeyError("Missing encoder query_init predictions in eval file.")

    selected = pred_prob >= first_hit_threshold if pred_first is None else pred_first.astype(bool)

    query_mask = _read_array(group, "targets/query_mask", required=False)
    n_keep = int(query_mask.astype(bool).sum()) if query_mask is not None else None
    if n_keep is None:
        return selected.astype(bool)

    selected_idx = np.flatnonzero(selected)
    if selected_idx.size == n_keep:
        return selected.astype(bool)

    total_hits = selected.shape[0]
    n_keep = min(n_keep, total_hits)
    out = np.zeros(total_hits, dtype=bool)

    if selected_idx.size == 0 and pred_prob is not None and n_keep > 0:
        keep = np.argsort(pred_prob)[::-1][:n_keep]
        out[keep] = True
        return out

    if selected_idx.size > n_keep:
        if pred_prob is not None:
            local_probs = pred_prob[selected_idx]
            keep_local = np.argsort(local_probs)[::-1][:n_keep]
            keep = selected_idx[keep_local]
            out[keep] = True
            return out
        out[selected_idx[:n_keep]] = True
        return out

    if selected_idx.size < n_keep and pred_prob is not None:
        keep = np.argsort(pred_prob)[::-1][:n_keep]
        out[keep] = True
        return out

    out[selected_idx] = True
    return out


def _counts_from_query_init_hits(
    selected_hits: np.ndarray,
    hit_owner: np.ndarray,
    n_particles: int,
) -> np.ndarray:
    counts = np.zeros(n_particles, dtype=np.int32)
    if n_particles == 0:
        return counts

    num_hits = min(selected_hits.shape[0], hit_owner.shape[0])
    if num_hits == 0:
        return counts

    selected_idx = np.flatnonzero(selected_hits[:num_hits])
    if selected_idx.size == 0:
        return counts

    owners = hit_owner[:num_hits][selected_idx]
    owners = owners[owners >= 0]

    if owners.size > 0:
        counts += np.bincount(owners, minlength=n_particles).astype(np.int32)
    return counts


def _extract_event_counts(
    group: h5py.Group,
    query_source: QuerySource,
    first_hit_threshold: float,
) -> ParticleCounts:
    valid_eta, valid_pt, n_hits, hit_owner, n_particles = _extract_truth_arrays(group)

    if query_source == QUERY_FINAL:
        n_queries = _counts_from_final_preds(group, hit_owner=hit_owner, n_particles=n_particles)
    elif query_source == QUERY_INIT_PRED:
        selected_hits = _select_hits_for_query_init(group, first_hit_threshold=first_hit_threshold)
        n_queries = _counts_from_query_init_hits(selected_hits, hit_owner=hit_owner, n_particles=n_particles)
    elif query_source == QUERY_INIT_TRUTH:
        selected_hits = _read_array(group, "targets/hit_is_first").astype(bool)
        n_queries = _counts_from_query_init_hits(selected_hits, hit_owner=hit_owner, n_particles=n_particles)
    else:
        raise ValueError(f"Unknown query source: {query_source}")

    return ParticleCounts(
        eta=valid_eta,
        pt=valid_pt,
        n_hits=n_hits,
        n_queries=n_queries,
    )


def _profile_stats(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, _ = np.histogram(x, bins=edges)
    sum_y, _ = np.histogram(x, bins=edges, weights=y)
    sum_y2, _ = np.histogram(x, bins=edges, weights=y * y)

    mean = np.divide(sum_y, counts, out=np.full_like(sum_y, np.nan, dtype=np.float64), where=counts > 0)
    var = np.divide(sum_y2, counts, out=np.zeros_like(sum_y2, dtype=np.float64), where=counts > 0) - mean * mean
    var = np.clip(var, a_min=0.0, a_max=None)
    err = np.divide(np.sqrt(var), np.sqrt(counts), out=np.full_like(mean, np.nan), where=counts > 0)
    return counts, mean, err


def _plot_hist(values: np.ndarray, xlabel: str, title: str, output: Path) -> None:
    if values.size == 0:
        raise ValueError(f"No values provided for histogram: {title}")

    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    # Integer-width bins can look "empty" when the range is large (e.g. 0..2200).
    # Use adaptive binning for wide ranges.
    if np.all(values == values.astype(np.int64)) and (vmax - vmin) <= 120:
        bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    else:
        n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
        bins = np.array([vmin - 0.5, vmax + 0.5], dtype=np.float64) if vmax == vmin else np.linspace(vmin, vmax, n_bins + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_facecolor("#f6f8fb")
    n, edges, _ = ax.hist(
        values,
        bins=bins,
        alpha=0.9,
        color="#2a77b4",
        edgecolor="#0f3557",
        linewidth=0.8,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    if n.size >= 3:
        smooth = np.convolve(n, np.ones(3) / 3.0, mode="same")
        ax.plot(centers, smooth, color="#0f3557", linewidth=1.4, alpha=0.95)
    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(mean, color="#c62828", linestyle="--", linewidth=1.4, label=f"mean={mean:.1f}")
    ax.axvline(median, color="#6a1b9a", linestyle=":", linewidth=1.4, label=f"median={median:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_event_scatter(
    particles_per_event: np.ndarray,
    hits_per_event: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_facecolor("#f6f8fb")
    hb = ax.hexbin(
        particles_per_event,
        hits_per_event,
        gridsize=42,
        mincnt=1,
        cmap="viridis",
        linewidths=0.15,
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("events per bin")
    ax.set_xlabel("particles per event (from hit_particle_id)")
    ax.set_ylabel("hits per event")
    ax.set_title("Hits per event vs particles per event")
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_profile(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    log_x: bool = False,
) -> None:
    counts, mean, err = _profile_stats(x, y.astype(np.float64), edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers[valid], mean[valid], yerr=err[valid], fmt="o-", capsize=2)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_eta_pt_heatmap(
    abs_eta: np.ndarray,
    pt: np.ndarray,
    values: np.ndarray,
    eta_edges: np.ndarray,
    pt_edges: np.ndarray,
    title: str,
    cbar_label: str,
    output: Path,
) -> None:
    sum_map, _, _ = np.histogram2d(abs_eta, pt, bins=[eta_edges, pt_edges], weights=values)
    cnt_map, _, _ = np.histogram2d(abs_eta, pt, bins=[eta_edges, pt_edges])
    mean_map = np.divide(sum_map, cnt_map, out=np.full_like(sum_map, np.nan, dtype=np.float64), where=cnt_map > 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        eta_edges,
        pt_edges,
        np.ma.masked_invalid(mean_map.T),
        shading="auto",
        cmap="viridis",
    )
    ax.set_xlabel("|eta|")
    ax.set_ylabel("pT [GeV]")
    ax.set_yscale("log")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_summary(
    output_path: Path,
    n_hits: np.ndarray,
    n_queries: np.ndarray,
    n_particles_per_event: np.ndarray,
    n_particles_per_event_hit_ids: np.ndarray,
    n_particles_over_cap: np.ndarray,
    n_hits_per_event: np.ndarray,
    eta: np.ndarray,
    pt: np.ndarray,
    eta_edges: np.ndarray,
    pt_edges: np.ndarray,
) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_particles", len(n_hits)])
        writer.writerow(["num_events", len(n_particles_per_event)])
        writer.writerow(["mean_particles_per_event_capped", float(np.mean(n_particles_per_event))])
        writer.writerow(["mean_hits_per_event", float(np.mean(n_hits_per_event))])
        if n_particles_per_event_hit_ids.size > 0:
            writer.writerow(["mean_particles_per_event_hit_ids", float(np.mean(n_particles_per_event_hit_ids))])
            writer.writerow(["mean_particles_over_cap", float(np.mean(n_particles_over_cap))])
        writer.writerow(["mean_hits_per_particle", float(np.mean(n_hits))])
        writer.writerow(["mean_queries_per_particle", float(np.mean(n_queries))])
        writer.writerow(["frac_particles_with_zero_queries", float(np.mean(n_queries == 0))])
        writer.writerow(["frac_particles_with_multi_queries", float(np.mean(n_queries > 1))])
        writer.writerow([])
        writer.writerow(["eta_bin_low", "eta_bin_high", "mean_hits", "mean_queries", "num_particles"])

        abs_eta = np.abs(eta)
        for lo, hi in pairwise(eta_edges):
            in_bin = (abs_eta >= lo) & (abs_eta < hi)
            if not np.any(in_bin):
                writer.writerow([float(lo), float(hi), "", "", 0])
            else:
                writer.writerow([float(lo), float(hi), float(np.mean(n_hits[in_bin])), float(np.mean(n_queries[in_bin])), int(in_bin.sum())])

        writer.writerow([])
        writer.writerow(["pt_bin_low", "pt_bin_high", "mean_hits", "mean_queries", "num_particles"])
        for lo, hi in pairwise(pt_edges):
            in_bin = (pt >= lo) & (pt < hi)
            if not np.any(in_bin):
                writer.writerow([float(lo), float(hi), "", "", 0])
            else:
                writer.writerow([float(lo), float(hi), float(np.mean(n_hits[in_bin])), float(np.mean(n_queries[in_bin])), int(in_bin.sum())])


def _count_unique_hit_particle_ids(hit_particle_id: np.ndarray) -> int:
    if hit_particle_id.size == 0:
        return 0

    ids = np.asarray(hit_particle_id)
    if np.issubdtype(ids.dtype, np.floating):
        ids = ids[np.isfinite(ids)]

    ids = ids[ids != -999]
    if ids.size == 0:
        return 0
    return int(np.unique(ids).size)


def _extract_event_particle_counts(group: h5py.Group) -> tuple[int, int, int]:
    """Return capped particles, hit-id particles, and overflow over cap for one event."""
    particle_valid = _read_array(group, "targets/particle_valid").astype(bool)
    n_capped = int(particle_valid.sum())

    hit_particle_id = _read_array(group, "targets/hit_particle_id", required=False)
    if hit_particle_id is None:
        return n_capped, n_capped, 0

    n_hit_ids = _count_unique_hit_particle_ids(np.asarray(hit_particle_id))
    return n_capped, n_hit_ids, max(0, n_hit_ids - n_capped)


def _extract_event_hit_count(group: h5py.Group) -> int:
    hit_valid = _read_array(group, "targets/hit_valid", required=False)
    if hit_valid is not None:
        return int(np.asarray(hit_valid).shape[0])

    hit_particle_id = _read_array(group, "targets/hit_particle_id", required=False)
    if hit_particle_id is not None:
        return int(np.asarray(hit_particle_id).shape[0])

    particle_hit_valid = _read_array(group, "targets/particle_hit_valid", required=False)
    if particle_hit_valid is not None:
        return int(np.asarray(particle_hit_valid).shape[-1])

    raise KeyError(f"Could not infer hit count for event '{group.name}' from targets.")


def main() -> None:
    settings = PlotConfig(
        # Required:
        eval_h5=Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-3l-fast-fix-vec-new-filter_20260227-T133500/ckpts/epoch=009-val_loss=1.79219_test_eval.h5"),
        # Optional:
        config=Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4.yaml"),
        out_dir=None,  # If None, writes to <eval_h5_parent>/particle_query_plots
        num_events=-1,  # -1 means all events in file
        query_source=QUERY_AUTO,  # one of: auto, final, query_init_pred, query_init_truth
        first_hit_threshold=None,  # If None, read from config query_init threshold (fallback 0.05)
        eta_max=None,  # If None, use config particle_max_abs_eta or data max
        pt_min=None,  # If None, use config particle_min_pt or data min>0
        pt_max=None,  # If None, use data 99.5 percentile
        n_eta_bins=10,
        n_pt_bins=12,
    )

    if not settings.eval_h5.exists():
        raise FileNotFoundError(f"Eval file does not exist: {settings.eval_h5}")

    cfg_data = {}
    if settings.config is not None:
        cfg_data = _read_yaml(settings.config).get("data", {})

    first_hit_threshold = settings.first_hit_threshold
    if first_hit_threshold is None:
        first_hit_threshold = 0.05
        if settings.config is not None:
            cfg = _read_yaml(settings.config)
            encoder_tasks = (
                cfg.get("model", {})
                .get("model", {})
                .get("init_args", {})
                .get("encoder_tasks", {})
                .get("init_args", {})
                .get("modules", [])
            )
            for task in encoder_tasks:
                init_args = task.get("init_args", {})
                if init_args.get("name") == "query_init":
                    first_hit_threshold = float(init_args.get("threshold", first_hit_threshold))
                    break

    out_dir = settings.out_dir if settings.out_dir is not None else settings.eval_h5.parent / "particle_query_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_eta = []
    all_pt = []
    all_hits = []
    all_queries = []
    all_particles_per_event_capped = []
    all_particles_per_event_hit_ids = []
    all_particles_over_cap = []
    all_hits_per_event = []

    with h5py.File(settings.eval_h5, "r") as h5f:
        event_keys = _sorted_event_keys(h5f)
        if settings.num_events > 0:
            event_keys = event_keys[: settings.num_events]
        if len(event_keys) == 0:
            raise RuntimeError("No events found in eval file.")

        query_source = _resolve_query_source(h5f, event_keys, settings.query_source)
        print(f"Using query source: {query_source}")
        t0 = time.perf_counter()

        for i, key in enumerate(event_keys):
            group = h5f[key]
            n_capped, n_hit_ids, n_over_cap = _extract_event_particle_counts(group)
            n_hits_event = _extract_event_hit_count(group)
            all_particles_per_event_capped.append(n_capped)
            all_particles_per_event_hit_ids.append(n_hit_ids)
            all_particles_over_cap.append(n_over_cap)
            all_hits_per_event.append(n_hits_event)

            counts = _extract_event_counts(
                group,
                query_source=query_source,
                first_hit_threshold=first_hit_threshold,
            )
            if counts.n_hits.size == 0:
                continue
            all_eta.append(counts.eta)
            all_pt.append(counts.pt)
            all_hits.append(counts.n_hits)
            all_queries.append(counts.n_queries)

            if (i + 1) % 50 == 0 or (i + 1) == len(event_keys):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                print(f"Processed {i + 1}/{len(event_keys)} events ({rate:.2f} events/s)")

    eta = np.concatenate(all_eta) if all_eta else np.array([], dtype=np.float64)
    pt = np.concatenate(all_pt) if all_pt else np.array([], dtype=np.float64)
    n_hits = np.concatenate(all_hits) if all_hits else np.array([], dtype=np.int32)
    n_queries = np.concatenate(all_queries) if all_queries else np.array([], dtype=np.int32)
    n_particles_per_event = np.asarray(all_particles_per_event_capped, dtype=np.int32)
    n_particles_per_event_hit_ids = np.asarray(all_particles_per_event_hit_ids, dtype=np.int32)
    n_particles_over_cap = np.asarray(all_particles_over_cap, dtype=np.int32)
    n_hits_per_event = np.asarray(all_hits_per_event, dtype=np.int32)

    if n_particles_per_event.size > 0:
        print(
            "Particles/event stats (capped):"
            f" min={int(np.min(n_particles_per_event))},"
            f" max={int(np.max(n_particles_per_event))},"
            f" mean={float(np.mean(n_particles_per_event)):.2f}"
        )
        print(
            "Particles/event stats (from hit_particle_id):"
            f" min={int(np.min(n_particles_per_event_hit_ids))},"
            f" max={int(np.max(n_particles_per_event_hit_ids))},"
            f" mean={float(np.mean(n_particles_per_event_hit_ids)):.2f},"
            f" mean_over_cap={float(np.mean(n_particles_over_cap)):.2f}"
        )
        print(
            "Hits/event stats:"
            f" min={int(np.min(n_hits_per_event))},"
            f" max={int(np.max(n_hits_per_event))},"
            f" mean={float(np.mean(n_hits_per_event)):.2f}"
        )

    if n_hits.size == 0:
        raise RuntimeError("No valid particles were found to plot.")

    eta_max = (
        float(settings.eta_max) if settings.eta_max is not None else float(cfg_data.get("particle_max_abs_eta", np.nanmax(np.abs(eta))))
    )
    pt_min = (
        float(settings.pt_min)
        if settings.pt_min is not None
        else float(cfg_data.get("particle_min_pt", max(np.nanmin(pt[pt > 0]), 1e-3)))
    )
    if settings.pt_max is not None:
        pt_max = float(settings.pt_max)
    else:
        pt_max = float(np.nanpercentile(pt, 99.5))
        pt_max = max(pt_max, pt_min * 1.5)

    eta_edges = np.linspace(0.0, eta_max, settings.n_eta_bins + 1)
    pt_edges = np.geomspace(pt_min, pt_max, settings.n_pt_bins + 1)
    abs_eta = np.abs(eta)

    _plot_hist(
        n_particles_per_event_hit_ids,
        xlabel="particles per event (from hit_particle_id)",
        title="Distribution of particles per event (from hit IDs)",
        output=out_dir / "particles_per_event_hist.png",
    )
    _plot_hist(
        n_hits_per_event,
        xlabel="hits per event",
        title="Distribution of hits per event",
        output=out_dir / "hits_per_event_hist.png",
    )
    _plot_event_scatter(
        particles_per_event=n_particles_per_event_hit_ids,
        hits_per_event=n_hits_per_event,
        output=out_dir / "hits_vs_particles_per_event_hexbin.png",
    )
    _plot_hist(
        n_particles_per_event,
        xlabel="truth particles per event (capped by event_max_num_particles)",
        title="Distribution of particles per event (capped)",
        output=out_dir / "particles_per_event_capped_hist.png",
    )
    _plot_hist(
        n_particles_over_cap,
        xlabel="particles over cap per event",
        title="Distribution of particles over cap per event",
        output=out_dir / "particles_over_cap_hist.png",
    )

    _plot_hist(
        n_hits,
        xlabel="truth hits per particle",
        title="Distribution of hits per truth particle",
        output=out_dir / "hits_per_particle_hist.png",
    )
    _plot_profile(
        abs_eta,
        n_hits,
        edges=eta_edges,
        xlabel="|eta|",
        ylabel="mean hits per particle",
        title="Hits per particle vs |eta|",
        output=out_dir / "hits_per_particle_vs_abs_eta.png",
    )
    _plot_profile(
        pt,
        n_hits,
        edges=pt_edges,
        xlabel="pT [GeV]",
        ylabel="mean hits per particle",
        title="Hits per particle vs pT",
        output=out_dir / "hits_per_particle_vs_pt.png",
        log_x=True,
    )
    _plot_eta_pt_heatmap(
        abs_eta,
        pt,
        n_hits,
        eta_edges=eta_edges,
        pt_edges=pt_edges,
        title="Mean hits per particle in (|eta|, pT) bins",
        cbar_label="mean hits",
        output=out_dir / "hits_per_particle_eta_pt_heatmap.png",
    )

    _plot_hist(
        n_queries,
        xlabel="queries per truth particle",
        title="Distribution of queries per truth particle",
        output=out_dir / "queries_per_particle_hist.png",
    )
    _plot_profile(
        abs_eta,
        n_queries,
        edges=eta_edges,
        xlabel="|eta|",
        ylabel="mean queries per particle",
        title="Queries per particle vs |eta|",
        output=out_dir / "queries_per_particle_vs_abs_eta.png",
    )
    _plot_profile(
        pt,
        n_queries,
        edges=pt_edges,
        xlabel="pT [GeV]",
        ylabel="mean queries per particle",
        title="Queries per particle vs pT",
        output=out_dir / "queries_per_particle_vs_pt.png",
        log_x=True,
    )
    _plot_eta_pt_heatmap(
        abs_eta,
        pt,
        n_queries,
        eta_edges=eta_edges,
        pt_edges=pt_edges,
        title="Mean queries per particle in (|eta|, pT) bins",
        cbar_label="mean queries",
        output=out_dir / "queries_per_particle_eta_pt_heatmap.png",
    )

    _write_summary(
        output_path=out_dir / "summary.csv",
        n_hits=n_hits,
        n_queries=n_queries,
        n_particles_per_event=n_particles_per_event,
        n_particles_per_event_hit_ids=n_particles_per_event_hit_ids,
        n_particles_over_cap=n_particles_over_cap,
        n_hits_per_event=n_hits_per_event,
        eta=eta,
        pt=pt,
        eta_edges=eta_edges,
        pt_edges=pt_edges,
    )

    print(f"Saved plots to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
