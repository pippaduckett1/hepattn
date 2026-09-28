# """Plot per-event TrackML efficiency and fake rate against event hit multiplicity.

# Efficiency uses the model-aligned 50% hit-overlap working point. Fake rate uses
# the DM definition from ``eval_utils_simple.py``: the fraction of included valid
# tracks that fail either majority-particle precision or recall above 50%.

# This script is self-contained and can be copied beside a completed standard
# ``*_test_eval.h5`` file. Install its only dependencies with:

#     python -m pip install h5py matplotlib numpy

# Then run, for example:

#     python plot_event_metrics_vs_hits.py epoch=029-val_loss=0.28273_test_eval.h5

# The exported ``event_metrics_vs_hits.csv`` can also be passed to regenerate the
# plots without rereading the evaluation HDF5 file.
# """

# from __future__ import annotations

# import argparse
# import csv
# from dataclasses import dataclass
# from pathlib import Path

# import h5py
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# import numpy as np


# WORKING_POINTS = (0.5,)
# NUM_HIT_BINS = 5


# @dataclass
# class EventMetrics:
#     event_id: str
#     num_hits: int
#     values: dict[float, tuple[float, float]]


# def _read_sample(group: h5py.Group, path: str) -> np.ndarray:
#     array = np.asarray(group[path])
#     return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


# def _dm_fake_rate(pred_valid: np.ndarray, pred_masks: np.ndarray, hit_particle_id: np.ndarray, hit_valid: np.ndarray) -> float:
#     """Return the old evaluator's DM fake rate for one standard-format event."""
#     valid_tracks = pred_valid & (pred_masks[:, hit_valid].sum(axis=-1) >= 3)
#     valid_indices = np.flatnonzero(valid_tracks)
#     if valid_indices.size == 0:
#         return float("nan")

#     hit_particle_id = hit_particle_id[hit_valid]
#     selected_masks = pred_masks[valid_indices][:, hit_valid]
#     dm_match = np.zeros(valid_indices.size, dtype=bool)
#     identical_duplicate = np.zeros(valid_indices.size, dtype=bool)
#     majority_particle_id = np.full(valid_indices.size, -1, dtype=np.int64)
#     seen_masks: set[bytes] = set()

#     for track_index, mask in enumerate(selected_masks):
#         mask_bytes = mask.tobytes()
#         if mask_bytes in seen_masks:
#             identical_duplicate[track_index] = True
#         else:
#             seen_masks.add(mask_bytes)

#         assigned_particle_ids = hit_particle_id[mask]
#         particle_ids, counts = np.unique(assigned_particle_ids, return_counts=True)
#         majority_index = int(np.argmax(counts))
#         particle_id = int(particle_ids[majority_index])
#         matched_hits = int(counts[majority_index])
#         majority_particle_id[track_index] = particle_id
#         precision = matched_hits / int(mask.sum())
#         recall = matched_hits / max(int(np.count_nonzero(hit_particle_id == particle_id)), 1)
#         dm_match[track_index] = (precision > 0.5) and (recall > 0.5)

#     included = ~identical_duplicate
#     for particle_id in np.unique(majority_particle_id):
#         if particle_id <= 0:
#             continue
#         same_particle = np.flatnonzero((majority_particle_id == particle_id) & included)
#         successful = same_particle[dm_match[same_particle]]
#         if successful.size:
#             dm_match[same_particle] = False
#             dm_match[successful[0]] = True

#     return float((~dm_match[included]).mean()) if included.any() else float("nan")


# def _event_metrics(group: h5py.Group) -> EventMetrics:
#     true_valid = _read_sample(group, "targets/particle_valid").astype(bool)
#     query_mask = _read_sample(group, "targets/query_mask").astype(bool)
#     pred_valid = _read_sample(group, "preds/final/track_valid/track_valid").astype(bool) & query_mask
#     pred_masks = _read_sample(group, "preds/final/track_hit_valid/track_hit_valid").astype(bool)
#     true_masks = _read_sample(group, "targets/particle_hit_valid").astype(bool)

#     true_masks &= true_valid[:, None]
#     active_pred_masks = pred_masks & pred_valid[:, None]
#     hit_tp = (active_pred_masks & true_masks).sum(axis=-1)
#     hit_t = true_masks.sum(axis=-1)
#     both_valid = true_valid & pred_valid
#     hit_valid = _read_sample(group, "targets/hit_valid").astype(bool)
#     hit_particle_id = _read_sample(group, "targets/hit_particle_id")
#     fake_rate = _dm_fake_rate(pred_valid, pred_masks, hit_particle_id, hit_valid)

#     values = {}
#     for working_point in WORKING_POINTS:
#         eff_pass = both_valid & (hit_tp / np.maximum(hit_t, 1) >= working_point)
#         efficiency = float(eff_pass.sum() / true_valid.sum()) if true_valid.any() else float("nan")
#         values[working_point] = (efficiency, fake_rate)

#     return EventMetrics(event_id=group.name.rsplit("/", maxsplit=1)[-1], num_hits=int(hit_valid.sum()), values=values)


# def _event_keys(h5_file: h5py.File) -> list[str]:
#     return sorted(h5_file.keys(), key=lambda key: int(key) if key.isdigit() else key)


# def _read_events_csv(path: Path) -> list[EventMetrics]:
#     events = []
#     with path.open(newline="") as file:
#         for row in csv.DictReader(file):
#             values = {
#                 working_point: (
#                     float(row[f"p{working_point:g}_efficiency"]),
#                     float(row[f"p{working_point:g}_fake_rate"]),
#                 )
#                 for working_point in WORKING_POINTS
#             }
#             events.append(EventMetrics(event_id=row["event_id"], num_hits=int(row["num_hits"]), values=values))
#     if not events:
#         raise ValueError(f"No event rows found in {path}.")
#     return events


# def _plot(events: list[EventMetrics], output: Path) -> None:
#     hit_counts = np.array([event.num_hits for event in events])
#     fig, axes = plt.subplots(len(WORKING_POINTS), 2, figsize=(12, 3.5), sharex=True, sharey="col", layout="constrained", squeeze=False)
#     for row, working_point in enumerate(WORKING_POINTS):
#         for col, (metric, label) in enumerate(((0, "Efficiency [%]"), (1, "DM fake rate [%]"))):
#             axis = axes[row, col]
#             values = np.array([event.values[working_point][metric] for event in events])
#             axis.scatter(hit_counts, values, s=22, alpha=0.6, color="#1f77b4")
#             if metric == 0:
#                 axis.set_ylim(0.9, 1.0)
#             else:
#                 axis.set_ylim(0.0, 0.02)
#             axis.set_ylabel(label)
#             axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0 if metric == 0 else 2))
#             axis.grid(alpha=0.25, linestyle="--")
#     for axis in axes[-1]:
#         axis.set_xlabel("Number of hits in event")
#     fig.savefig(output, dpi=200)
#     plt.close(fig)


# def _bin_event_metrics(
#     events: list[EventMetrics], num_bins: int
# ) -> tuple[np.ndarray, dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
#     hit_counts = np.array([event.num_hits for event in events])
#     edges = np.linspace(hit_counts.min(), hit_counts.max(), num_bins + 1)
#     bin_indices = np.clip(np.digitize(hit_counts, edges[1:-1]), 0, num_bins - 1)
#     stats = {}
#     for working_point in WORKING_POINTS:
#         stats[working_point] = {}
#         for metric in (0, 1):
#             values = np.array([event.values[working_point][metric] for event in events])
#             means = np.full(num_bins, np.nan)
#             errors = np.full(num_bins, np.nan)
#             counts = np.zeros(num_bins, dtype=int)
#             for bin_index in range(num_bins):
#                 selected = values[bin_indices == bin_index]
#                 selected = selected[np.isfinite(selected)]
#                 counts[bin_index] = len(selected)
#                 if len(selected):
#                     means[bin_index] = selected.mean()
#                     errors[bin_index] = selected.std(ddof=1) / np.sqrt(len(selected)) if len(selected) > 1 else 0.0
#             stats[working_point][metric] = (means, errors, counts)
#     return edges, stats


# def _plot_binned_events(
#     events: list[EventMetrics], output: Path, num_bins: int
# ) -> tuple[np.ndarray, dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
#     edges, stats = _bin_event_metrics(events, num_bins)
#     centers = 0.5 * (edges[:-1] + edges[1:])
#     fig, axes = plt.subplots(len(WORKING_POINTS), 2, figsize=(12, 3.5), sharex=True, sharey="col", layout="constrained", squeeze=False)
#     for row, working_point in enumerate(WORKING_POINTS):
#         for col, (metric, label) in enumerate(((0, "Efficiency [%]"), (1, "DM fake rate [%]"))):
#             axis = axes[row, col]
#             means, errors, counts = stats[working_point][metric]
#             valid = counts > 0
#             axis.errorbar(centers[valid], means[valid], yerr=errors[valid], fmt="o-", color="#b22222", capsize=3)
#             if metric == 0:
#                 axis.set_ylim(0.9, 1.0)
#             else:
#                 axis.set_ylim(0.0, 0.02)
#             axis.set_ylabel(label)
#             axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0 if metric == 0 else 2))
#             axis.grid(alpha=0.25, linestyle="--")
#     for axis in axes[-1]:
#         axis.set_xlabel(f"Number of hits in event ({num_bins} equal-width bins)")
#     fig.savefig(output, dpi=200)
#     plt.close(fig)
#     return edges, stats


# def _write_csv(events: list[EventMetrics], output: Path) -> None:
#     fields = ["event_id", "num_hits"] + [f"p{working_point:g}_{name}" for working_point in WORKING_POINTS for name in ("efficiency", "fake_rate")]
#     with output.open("w", newline="") as file:
#         writer = csv.DictWriter(file, fieldnames=fields)
#         writer.writeheader()
#         for event in events:
#             row = {"event_id": event.event_id, "num_hits": event.num_hits}
#             for working_point, (efficiency, fake_rate) in event.values.items():
#                 row[f"p{working_point:g}_efficiency"] = efficiency
#                 row[f"p{working_point:g}_fake_rate"] = fake_rate
#             writer.writerow(row)


# def _write_binned_csv(
#     edges: np.ndarray,
#     stats: dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
#     output: Path,
# ) -> None:
#     fields = ["hit_bin_low", "hit_bin_high", "num_events"] + [
#         f"p{working_point:g}_{name}" for working_point in WORKING_POINTS for name in ("efficiency", "efficiency_sem", "fake_rate", "fake_rate_sem")
#     ]
#     with output.open("w", newline="") as file:
#         writer = csv.DictWriter(file, fieldnames=fields)
#         writer.writeheader()
#         for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
#             row = {"hit_bin_low": low, "hit_bin_high": high, "num_events": int(stats[WORKING_POINTS[0]][0][2][bin_index])}
#             for working_point in WORKING_POINTS:
#                 eff_mean, eff_error, _ = stats[working_point][0]
#                 fake_mean, fake_error, _ = stats[working_point][1]
#                 row[f"p{working_point:g}_efficiency"] = eff_mean[bin_index]
#                 row[f"p{working_point:g}_efficiency_sem"] = eff_error[bin_index]
#                 row[f"p{working_point:g}_fake_rate"] = fake_mean[bin_index]
#                 row[f"p{working_point:g}_fake_rate_sem"] = fake_error[bin_index]
#             writer.writerow(row)


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("input_path", type=Path, help="Standard *_test_eval.h5 output or an exported event_metrics_vs_hits.csv.")
#     parser.add_argument("--out-dir", type=Path, default=None, help="Defaults beside the input; CSV input defaults to its parent directory.")
#     parser.add_argument("--num-hit-bins", type=int, default=NUM_HIT_BINS, help="Number of equal-width event-hit bins for the aggregate plot.")
#     args = parser.parse_args()

#     if args.num_hit_bins < 1:
#         raise ValueError("--num-hit-bins must be positive.")
#     if args.input_path.suffix == ".csv":
#         events = _read_events_csv(args.input_path)
#         print(f"Read {len(events)} events from {args.input_path}")
#         default_out_dir = args.input_path.parent
#     else:
#         with h5py.File(args.input_path, "r") as h5_file:
#             keys = _event_keys(h5_file)
#             if not keys:
#                 raise ValueError(
#                     f"No event groups found in {args.input_path}. "
#                     "The evaluation HDF5 is empty or incomplete; wait for the prediction writer to finish before running this script."
#                 )
#             events = []
#             for index, key in enumerate(keys, start=1):
#                 events.append(_event_metrics(h5_file[key]))
#                 if index % 25 == 0 or index == len(keys):
#                     print(f"Processed {index}/{len(keys)} events")
#         default_out_dir = args.input_path.parent / f"{args.input_path.stem}_event_metrics_vs_hits"

#     out_dir = args.out_dir or default_out_dir
#     out_dir.mkdir(parents=True, exist_ok=True)
#     csv_output = out_dir / "event_metrics_vs_hits.csv"
#     reused_input_csv = csv_output.resolve() == args.input_path.resolve()
#     if not reused_input_csv:
#         _write_csv(events, csv_output)
#     _plot(events, out_dir / "event_metrics_vs_hits.png")
#     edges, stats = _plot_binned_events(events, out_dir / f"event_metrics_vs_hits_binned_{args.num_hit_bins}.png", args.num_hit_bins)
#     _write_binned_csv(edges, stats, out_dir / f"event_metrics_vs_hits_binned_{args.num_hit_bins}.csv")
#     print(f"Reused {csv_output}" if reused_input_csv else f"Wrote {csv_output}")
#     print(f"Wrote {out_dir / 'event_metrics_vs_hits.png'}")
#     print(f"Wrote {out_dir / f'event_metrics_vs_hits_binned_{args.num_hit_bins}.csv'}")
#     print(f"Wrote {out_dir / f'event_metrics_vs_hits_binned_{args.num_hit_bins}.png'}")


# if __name__ == "__main__":
#     main()


"""Plot per-event TrackML efficiency and fake rate against event hit multiplicity.

Efficiency uses the model-aligned 50% hit-overlap working point. Fake rate uses
the DM definition from ``eval_utils_simple.py``: the fraction of included valid
tracks that fail either majority-particle precision or recall above 50%.

This script is self-contained and can be copied beside a completed paper-format
``__test.h5`` file produced by ``PredictionWriter``. Install its only
dependencies with:

    python -m pip install h5py matplotlib numpy

Then run, for example:

    python plot_event_metrics_vs_hits.py epoch=029-val_loss=0.28273__test.h5

The exported ``event_metrics_vs_hits.csv`` can also be passed to regenerate the
plots without rereading the evaluation HDF5 file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


WORKING_POINTS = (0.5,)
NUM_HIT_BINS = 5


@dataclass
class EventMetrics:
    event_id: str
    num_hits: int
    values: dict[float, tuple[float, float]]


def _paper_pred_valid(preds: h5py.Group, track_valid_threshold: float, iou_threshold: float) -> np.ndarray:
    """Load valid-track predictions from a paper-format event."""
    if "track_valid_prob" in preds:
        valid_prob = np.asarray(preds["track_valid_prob"], dtype=np.float32)
        pred_valid = np.isfinite(valid_prob) & (valid_prob >= track_valid_threshold)
    elif "class_preds" in preds:
        pred_valid = np.asarray(preds["class_preds"]).argmax(axis=-1) == 0
    else:
        raise KeyError(f"Missing track-valid prediction in {preds.name}.")

    if iou_threshold > 0:
        if "query_iou" not in preds:
            raise KeyError(f"--iou-threshold was set, but {preds.name}/query_iou is missing.")
        pred_iou = np.asarray(preds["query_iou"], dtype=np.float32)
        pred_valid &= np.isfinite(pred_iou) & (pred_iou >= iou_threshold)
    return pred_valid


def _dm_metrics(
    pred_valid: np.ndarray,
    pred_masks: np.ndarray,
    hit_particle_id: np.ndarray,
    reference_hits_by_pid: dict[int, int],
) -> tuple[float, set[int]]:
    """Return old-evaluator DM fake rate and the particle IDs reconstructed by DM tracks."""
    valid_tracks = pred_valid & (pred_masks.sum(axis=-1) >= 3)
    valid_indices = np.flatnonzero(valid_tracks)
    if valid_indices.size == 0:
        return float("nan"), set()

    selected_masks = pred_masks[valid_indices]
    dm_match = np.zeros(valid_indices.size, dtype=bool)
    identical_duplicate = np.zeros(valid_indices.size, dtype=bool)
    majority_particle_id = np.full(valid_indices.size, -1, dtype=np.int64)
    seen_masks: set[bytes] = set()

    for track_index, mask in enumerate(selected_masks):
        mask_bytes = mask.tobytes()
        if mask_bytes in seen_masks:
            identical_duplicate[track_index] = True
        else:
            seen_masks.add(mask_bytes)

        assigned_particle_ids = hit_particle_id[mask]
        particle_ids, counts = np.unique(assigned_particle_ids, return_counts=True)
        majority_index = int(np.argmax(counts))
        particle_id = int(particle_ids[majority_index])
        matched_hits = int(counts[majority_index])
        majority_particle_id[track_index] = particle_id
        precision = matched_hits / int(mask.sum())
        n_true_hits = reference_hits_by_pid.get(particle_id, int(np.count_nonzero(hit_particle_id == particle_id)))
        recall = matched_hits / max(n_true_hits, 1)
        dm_match[track_index] = (precision > 0.5) and (recall > 0.5)

    included = ~identical_duplicate
    for particle_id in np.unique(majority_particle_id):
        if particle_id <= 0:
            continue
        same_particle = np.flatnonzero((majority_particle_id == particle_id) & included)
        successful = same_particle[dm_match[same_particle]]
        if successful.size:
            dm_match[same_particle] = False
            dm_match[successful[0]] = True

    reconstructed_particle_ids = set(majority_particle_id[included & dm_match].tolist())
    fake_rate = float((~dm_match[included]).mean()) if included.any() else float("nan")
    return fake_rate, reconstructed_particle_ids


def _event_metrics(
    group: h5py.Group,
    track_valid_threshold: float,
    iou_threshold: float,
    eta_cut: float,
    pt_cut: float,
) -> EventMetrics:
    """Evaluate one ``PredictionWriter`` paper-format event group."""
    hit_particle_id = np.asarray(group["hits/pids"], dtype=np.int64)
    pred_masks = np.asarray(group["preds/masks"]) > 0
    if pred_masks.shape[-1] != hit_particle_id.size:
        raise ValueError(f"Hit-axis mismatch in {group.name}: masks have {pred_masks.shape[-1]} hits, labels have {hit_particle_id.size}.")

    part_ids = np.asarray(group["parts/pids"], dtype=np.int64)
    part_pt = np.asarray(group["parts/pts"], dtype=np.float32)
    part_eta = np.asarray(group["parts/etas"], dtype=np.float32)
    part_n_hits = np.asarray(group["parts/n_hits"], dtype=np.int64)
    if not (part_ids.size == part_pt.size == part_eta.size == part_n_hits.size):
        raise ValueError(f"Inconsistent particle arrays in {group.name}.")

    reference_hits_by_pid = dict(zip(part_ids.tolist(), part_n_hits.tolist(), strict=False))
    reconstructable = (part_n_hits >= 3) & (np.abs(part_eta) < eta_cut) & (part_pt > pt_cut)
    pred_valid = _paper_pred_valid(group["preds"], track_valid_threshold, iou_threshold)
    fake_rate, reconstructed_particle_ids = _dm_metrics(pred_valid, pred_masks, hit_particle_id, reference_hits_by_pid)
    efficiency = float(np.isin(part_ids[reconstructable], list(reconstructed_particle_ids)).mean()) if reconstructable.any() else float("nan")

    return EventMetrics(
        event_id=group.name.rsplit("/", maxsplit=1)[-1],
        num_hits=int(hit_particle_id.size),
        values={0.5: (efficiency, fake_rate)},
    )


def _event_keys(h5_file: h5py.File) -> list[str]:
    return sorted(
        (key for key in h5_file.keys() if key.startswith("event_")),
        key=lambda key: int(key.removeprefix("event_")),
    )


def _read_events_csv(path: Path) -> list[EventMetrics]:
    events = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            values = {
                working_point: (
                    float(row[f"p{working_point:g}_efficiency"]),
                    float(row[f"p{working_point:g}_fake_rate"]),
                )
                for working_point in WORKING_POINTS
            }
            events.append(EventMetrics(event_id=row["event_id"], num_hits=int(row["num_hits"]), values=values))
    if not events:
        raise ValueError(f"No event rows found in {path}.")
    return events


def _plot(events: list[EventMetrics], output: Path) -> None:
    hit_counts = np.array([event.num_hits for event in events])
    fig, axes = plt.subplots(len(WORKING_POINTS), 2, figsize=(12, 3.5), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for row, working_point in enumerate(WORKING_POINTS):
        for col, (metric, label) in enumerate(((0, "Efficiency [%]"), (1, "DM fake rate [%]"))):
            axis = axes[row, col]
            values = np.array([event.values[working_point][metric] for event in events])
            axis.scatter(hit_counts, values, s=22, alpha=0.6, color="#1f77b4")
            if metric == 0:
                axis.set_ylim(0.9, 1.0)
            else:
                axis.set_ylim(0.0, 0.02)
            axis.set_ylabel(label)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0 if metric == 0 else 2))
            axis.grid(alpha=0.25, linestyle="--")
    for axis in axes[-1]:
        axis.set_xlabel("Number of hits in event")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _bin_event_metrics(
    events: list[EventMetrics], num_bins: int
) -> tuple[np.ndarray, dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    hit_counts = np.array([event.num_hits for event in events])
    edges = np.linspace(hit_counts.min(), hit_counts.max(), num_bins + 1)
    bin_indices = np.clip(np.digitize(hit_counts, edges[1:-1]), 0, num_bins - 1)
    stats = {}
    for working_point in WORKING_POINTS:
        stats[working_point] = {}
        for metric in (0, 1):
            values = np.array([event.values[working_point][metric] for event in events])
            means = np.full(num_bins, np.nan)
            errors = np.full(num_bins, np.nan)
            counts = np.zeros(num_bins, dtype=int)
            for bin_index in range(num_bins):
                selected = values[bin_indices == bin_index]
                selected = selected[np.isfinite(selected)]
                counts[bin_index] = len(selected)
                if len(selected):
                    means[bin_index] = selected.mean()
                    errors[bin_index] = selected.std(ddof=1) / np.sqrt(len(selected)) if len(selected) > 1 else 0.0
            stats[working_point][metric] = (means, errors, counts)
    return edges, stats


def _plot_binned_events(
    events: list[EventMetrics], output: Path, num_bins: int
) -> tuple[np.ndarray, dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    edges, stats = _bin_event_metrics(events, num_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(len(WORKING_POINTS), 2, figsize=(12, 3.5), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for row, working_point in enumerate(WORKING_POINTS):
        for col, (metric, label) in enumerate(((0, "Efficiency [%]"), (1, "DM fake rate [%]"))):
            axis = axes[row, col]
            means, errors, counts = stats[working_point][metric]
            valid = counts > 0
            axis.errorbar(centers[valid], means[valid], yerr=errors[valid], fmt="o-", color="#b22222", capsize=3)
            if metric == 0:
                axis.set_ylim(0.9, 1.0)
            else:
                axis.set_ylim(0.0, 0.02)
            axis.set_ylabel(label)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0 if metric == 0 else 2))
            axis.grid(alpha=0.25, linestyle="--")
    for axis in axes[-1]:
        axis.set_xlabel(f"Number of hits in event ({num_bins} equal-width bins)")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return edges, stats


def _write_csv(events: list[EventMetrics], output: Path) -> None:
    fields = ["event_id", "num_hits"] + [f"p{working_point:g}_{name}" for working_point in WORKING_POINTS for name in ("efficiency", "fake_rate")]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = {"event_id": event.event_id, "num_hits": event.num_hits}
            for working_point, (efficiency, fake_rate) in event.values.items():
                row[f"p{working_point:g}_efficiency"] = efficiency
                row[f"p{working_point:g}_fake_rate"] = fake_rate
            writer.writerow(row)


def _write_binned_csv(
    edges: np.ndarray,
    stats: dict[float, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    output: Path,
) -> None:
    fields = ["hit_bin_low", "hit_bin_high", "num_events"] + [
        f"p{working_point:g}_{name}" for working_point in WORKING_POINTS for name in ("efficiency", "efficiency_sem", "fake_rate", "fake_rate_sem")
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            row = {"hit_bin_low": low, "hit_bin_high": high, "num_events": int(stats[WORKING_POINTS[0]][0][2][bin_index])}
            for working_point in WORKING_POINTS:
                eff_mean, eff_error, _ = stats[working_point][0]
                fake_mean, fake_error, _ = stats[working_point][1]
                row[f"p{working_point:g}_efficiency"] = eff_mean[bin_index]
                row[f"p{working_point:g}_efficiency_sem"] = eff_error[bin_index]
                row[f"p{working_point:g}_fake_rate"] = fake_mean[bin_index]
                row[f"p{working_point:g}_fake_rate_sem"] = fake_error[bin_index]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Paper-format __test.h5 output or an exported event_metrics_vs_hits.csv.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults beside the input; CSV input defaults to its parent directory.")
    parser.add_argument("--num-hit-bins", type=int, default=NUM_HIT_BINS, help="Number of equal-width event-hit bins for the aggregate plot.")
    parser.add_argument("--track-valid-threshold", type=float, default=0.5, help="Minimum paper-format track-valid probability.")
    parser.add_argument("--iou-threshold", type=float, default=0.0, help="Optional minimum predicted IoU.")
    parser.add_argument("--eta-cut", type=float, default=4.0, help="Maximum absolute truth-particle eta for efficiency.")
    parser.add_argument("--pt-cut", type=float, default=1.0, help="Minimum truth-particle pT for efficiency.")
    args = parser.parse_args()

    if args.num_hit_bins < 1:
        raise ValueError("--num-hit-bins must be positive.")
    if args.input_path.suffix == ".csv":
        events = _read_events_csv(args.input_path)
        print(f"Read {len(events)} events from {args.input_path}")
        default_out_dir = args.input_path.parent
    else:
        with h5py.File(args.input_path, "r") as h5_file:
            keys = _event_keys(h5_file)
            if not keys:
                raise ValueError(
                    f"No event groups found in {args.input_path}. "
                    "The evaluation HDF5 is empty or incomplete; wait for the prediction writer to finish before running this script."
                )
            events = []
            for index, key in enumerate(keys, start=1):
                events.append(
                    _event_metrics(
                        h5_file[key],
                        track_valid_threshold=args.track_valid_threshold,
                        iou_threshold=args.iou_threshold,
                        eta_cut=args.eta_cut,
                        pt_cut=args.pt_cut,
                    )
                )
                if index % 25 == 0 or index == len(keys):
                    print(f"Processed {index}/{len(keys)} events")
        default_out_dir = args.input_path.parent / f"{args.input_path.stem}_event_metrics_vs_hits"

    out_dir = args.out_dir or default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_output = out_dir / "event_metrics_vs_hits.csv"
    reused_input_csv = csv_output.resolve() == args.input_path.resolve()
    if not reused_input_csv:
        _write_csv(events, csv_output)
    _plot(events, out_dir / "event_metrics_vs_hits.png")
    edges, stats = _plot_binned_events(events, out_dir / f"event_metrics_vs_hits_binned_{args.num_hit_bins}.png", args.num_hit_bins)
    _write_binned_csv(edges, stats, out_dir / f"event_metrics_vs_hits_binned_{args.num_hit_bins}.csv")
    print(f"Reused {csv_output}" if reused_input_csv else f"Wrote {csv_output}")
    print(f"Wrote {out_dir / 'event_metrics_vs_hits.png'}")
    print(f"Wrote {out_dir / f'event_metrics_vs_hits_binned_{args.num_hit_bins}.csv'}")
    print(f"Wrote {out_dir / f'event_metrics_vs_hits_binned_{args.num_hit_bins}.png'}")


if __name__ == "__main__":
    main()
