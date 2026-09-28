# #!/usr/bin/env python3
# """Plot initialised-query counts against reconstructable truth particles per event."""

# from __future__ import annotations

# import argparse
# import csv
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path

# import h5py
# import numpy as np


# SRC_ROOT = Path(__file__).resolve().parents[4]
# if str(SRC_ROOT) not in sys.path:
#     sys.path.insert(0, str(SRC_ROOT))


# @dataclass
# class EventCounts:
#     sample_ids: list[str]
#     num_initialised_queries: np.ndarray
#     num_reconstructable_particles: np.ndarray
#     num_pre_filter_hits: np.ndarray

#     @property
#     def difference(self) -> np.ndarray:
#         return self.num_initialised_queries - self.num_reconstructable_particles


# @dataclass
# class LinearFit:
#     slope: float
#     intercept: float
#     r2: float


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Read a TrackML tracking eval HDF5 file and plot initialised queries per event "
#             "against reconstructable truth particles per event."
#         ),
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("--eval-h5", type=Path, required=True, help="Path to run_tracking.py test eval HDF5.")
#     parser.add_argument(
#         "--max-events",
#         type=int,
#         default=-1,
#         help="Maximum number of events to process. Use -1 for all available events.",
#     )
#     parser.add_argument(
#         "--add-linear-fit",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Overlay a least-squares linear fit on the scatter plot.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=Path,
#         default=None,
#         help="Directory where the figure and CSV summaries are written. Defaults to <eval_h5_parent>/query_init_event_plots.",
#     )
#     parser.add_argument(
#         "--output-stem",
#         type=str,
#         default="initialised_queries_vs_reconstructable_particles",
#         help="Filename stem used for output files.",
#     )
#     return parser.parse_args()


# def _get_pyplot():
#     try:
#         import matplotlib as mpl
#     except ModuleNotFoundError as exc:
#         raise ModuleNotFoundError(
#             "matplotlib is required to generate initialised-query plots. Install it in the active environment and rerun."
#         ) from exc

#     mpl.use("Agg", force=True)
#     mpl.rcParams.update(
#         {
#             "font.family": "DejaVu Serif",
#             "font.size": 16,
#             "axes.titlesize": 18,
#             "axes.labelsize": 17,
#             "xtick.labelsize": 14,
#             "ytick.labelsize": 14,
#             "axes.spines.top": False,
#             "axes.spines.right": False,
#             "figure.facecolor": "white",
#             "axes.facecolor": "#fcfcfc",
#             "savefig.facecolor": "white",
#         }
#     )
#     import matplotlib.pyplot as plt

#     return plt


# def _sorted_event_keys(h5f: h5py.File) -> list[str]:
#     def _key(key: str) -> tuple[int, int | str]:
#         return (0, int(key)) if key.isdigit() else (1, key)

#     return sorted(h5f.keys(), key=_key)


# def _read_array(group: h5py.Group, path: str, required: bool = True) -> np.ndarray | None:
#     try:
#         array = np.asarray(group[path])
#     except KeyError:
#         if required:
#             raise
#         return None
#     if array.ndim >= 1 and array.shape[0] == 1:
#         array = array[0]
#     return array


# def _read_scalar_int(group: h5py.Group, path: str, required: bool = True) -> int | None:
#     array = _read_array(group, path, required=required)
#     if array is None:
#         return None
#     return int(np.asarray(array).reshape(-1)[0])


# def _collect_event_counts(eval_h5: Path, max_events: int) -> EventCounts:
#     with h5py.File(eval_h5, "r") as h5f:
#         event_keys = _sorted_event_keys(h5f)
#         if max_events > 0:
#             event_keys = event_keys[:max_events]
#         if not event_keys:
#             raise RuntimeError(f"No events found in eval file: {eval_h5}")

#         n_events = len(event_keys)
#         num_initialised_queries = np.zeros(n_events, dtype=np.int32)
#         num_reconstructable_particles = np.zeros(n_events, dtype=np.int32)
#         num_pre_filter_hits = np.zeros(n_events, dtype=np.int32)

#         for i, key in enumerate(event_keys):
#             group = h5f[key]

#             init_queries = _read_scalar_int(group, "targets/num_initialised_queries", required=False)
#             if init_queries is None:
#                 query_mask = _read_array(group, "targets/query_mask", required=False)
#                 if query_mask is None:
#                     track_valid = _read_array(group, "preds/final/track_valid/track_valid", required=False)
#                     if track_valid is None:
#                         raise KeyError(
#                             "Eval file is missing both targets/num_initialised_queries and fallback query-count datasets."
#                         )
#                     init_queries = int(np.asarray(track_valid).astype(bool).shape[-1])
#                 else:
#                     init_queries = int(np.asarray(query_mask).astype(bool).sum())

#             reconstructable = _read_scalar_int(group, "targets/num_reconstructable_particles", required=False)
#             if reconstructable is None:
#                 particle_valid = _read_array(group, "targets/particle_valid")
#                 reconstructable = int(np.asarray(particle_valid).astype(bool).sum())

#             # Derive pre-filter hit count from particle_hit_valid shape (last dim = n_hits),
#             # falling back to summing all per-feature valid masks found under targets.
#             phv = _read_array(group, "targets/particle_hit_valid", required=False)
#             if phv is not None:
#                 pre_hits = int(np.asarray(phv).shape[-1])
#             else:
#                 pre_hits = 0
#                 if "targets" in group:
#                     for ds_name in group["targets"]:
#                         if ds_name.endswith("_valid") and ds_name not in (
#                             "particle_valid",
#                             "query_mask",
#                             "num_initialised_queries",
#                             "num_reconstructable_particles",
#                         ):
#                             arr = _read_array(group, f"targets/{ds_name}", required=False)
#                             if arr is not None and np.asarray(arr).ndim == 1:
#                                 pre_hits += int(np.asarray(arr).shape[0])

#             num_initialised_queries[i] = init_queries
#             num_reconstructable_particles[i] = reconstructable
#             num_pre_filter_hits[i] = pre_hits

#             if (i + 1) % 50 == 0 or (i + 1) == n_events:
#                 print(f"Processed {i + 1}/{n_events} events")

#     return EventCounts(
#         sample_ids=event_keys,
#         num_initialised_queries=num_initialised_queries,
#         num_reconstructable_particles=num_reconstructable_particles,
#         num_pre_filter_hits=num_pre_filter_hits,
#     )


# def _stats(values: np.ndarray) -> dict[str, float]:
#     values = values.astype(np.float64)
#     return {
#         "min": float(np.min(values)),
#         "mean": float(np.mean(values)),
#         "std": float(np.std(values)),
#         "max": float(np.max(values)),
#     }


# def _linear_fit(x: np.ndarray, y: np.ndarray) -> LinearFit | None:
#     x = x.astype(np.float64)
#     y = y.astype(np.float64)
#     if x.size < 2 or np.unique(x).size < 2:
#         return None

#     slope, intercept = np.polyfit(x, y, deg=1)
#     pred = intercept + slope * x
#     ss_res = float(np.square(y - pred).sum())
#     ss_tot = float(np.square(y - y.mean()).sum())
#     r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
#     return LinearFit(slope=float(slope), intercept=float(intercept), r2=float(r2))


# def _hist_bins(values: np.ndarray) -> np.ndarray:
#     if values.size == 1:
#         center = float(values[0])
#         return np.array([center - 0.5, center + 0.5], dtype=float)
#     if np.all(values == values.astype(np.int64)) and (values.max() - values.min()) <= 200:
#         return np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
#     n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
#     return np.linspace(values.min(), values.max(), n_bins + 1)


# def _plot_figure(
#     counts: EventCounts,
#     add_linear_fit: bool,
#     output_base: Path,
# ) -> LinearFit | None:
#     plt = _get_pyplot()
#     fit = _linear_fit(counts.num_reconstructable_particles, counts.num_initialised_queries) if add_linear_fit else None
#     hits_fit = _linear_fit(counts.num_pre_filter_hits, counts.num_reconstructable_particles) if add_linear_fit else None
#     diff = counts.difference

#     fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
#     scatter_ax, hist_ax, hits_ax = axes

#     x = counts.num_reconstructable_particles.astype(np.float64)
#     y = counts.num_initialised_queries.astype(np.float64)
#     scatter_ax.scatter(
#         x,
#         y,
#         s=20.0,
#         marker=".",
#         alpha=0.65,
#         color="#1f77b4",
#         edgecolors="none",
#         zorder=2,
#         rasterized=True,
#     )
#     if fit is not None:
#         x_line = np.linspace(float(x.min()), float(x.max()) * 1.02, 256)
#         scatter_ax.plot(
#             x_line,
#             fit.intercept + fit.slope * x_line,
#             color="#666666",
#             linestyle="--",
#             linewidth=0.9,
#             alpha=0.8,
#             zorder=3,
#         )
#         scatter_ax.text(
#             0.06,
#             0.94,
#             f"Slope = {fit.slope:.4f}\n$R^2$ = {fit.r2:.2f}",
#             transform=scatter_ax.transAxes,
#             ha="left",
#             va="top",
#             fontsize=14,
#             color="#222222",
#             bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
#         )

#     scatter_ax.set_xlabel("Reconstructable truth particles per event")
#     scatter_ax.set_ylabel("initialised queries per event")
#     scatter_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
#     scatter_ax.set_axisbelow(True)

#     diff_stats = _stats(diff)
#     bins = _hist_bins(diff)
#     hist_ax.hist(diff, bins=bins, alpha=0.85, color="#d62728", edgecolor="black", linewidth=0.35)
#     hist_ax.axvline(diff_stats["mean"], color="#444444", linestyle="--", linewidth=1.2)
#     hist_ax.set_xlabel("initialised queries - reconstructable particles")
#     hist_ax.set_ylabel("Events")
#     hist_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
#     hist_ax.set_axisbelow(True)
#     hist_ax.text(
#         0.97,
#         0.94,
#         f"Mean = {diff_stats['mean']:.2f}\nStd = {diff_stats['std']:.2f}",
#         transform=hist_ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=14,
#         color="#222222",
#         bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
#     )

#     hx = counts.num_pre_filter_hits.astype(np.float64)
#     hy = counts.num_reconstructable_particles.astype(np.float64)
#     hits_ax.scatter(
#         hx,
#         hy,
#         s=20.0,
#         marker=".",
#         alpha=0.65,
#         color="#2ca02c",
#         edgecolors="none",
#         zorder=2,
#         rasterized=True,
#     )
#     if hits_fit is not None:
#         hx_line = np.linspace(float(hx.min()), float(hx.max()) * 1.02, 256)
#         hits_ax.plot(
#             hx_line,
#             hits_fit.intercept + hits_fit.slope * hx_line,
#             color="#666666",
#             linestyle="--",
#             linewidth=0.9,
#             alpha=0.8,
#             zorder=3,
#         )
#         hits_ax.text(
#             0.06,
#             0.94,
#             f"Slope = {hits_fit.slope:.4f}\n$R^2$ = {hits_fit.r2:.2f}",
#             transform=hits_ax.transAxes,
#             ha="left",
#             va="top",
#             fontsize=14,
#             color="#222222",
#             bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
#         )
#     hits_ax.set_xlabel("Pre-filter hits per event")
#     hits_ax.set_ylabel("Reconstructable truth particles per event")
#     hits_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
#     hits_ax.set_axisbelow(True)

#     fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
#     fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     return fit


# def _write_per_event_csv(out_path: Path, counts: EventCounts) -> None:
#     with out_path.open("w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             [
#                 "event_index",
#                 "sample_id",
#                 "num_initialised_queries",
#                 "num_reconstructable_particles",
#                 "num_pre_filter_hits",
#                 "initialised_queries_minus_reconstructable_particles",
#             ]
#         )
#         for idx, sample_id in enumerate(counts.sample_ids):
#             writer.writerow(
#                 [
#                     idx,
#                     sample_id,
#                     int(counts.num_initialised_queries[idx]),
#                     int(counts.num_reconstructable_particles[idx]),
#                     int(counts.num_pre_filter_hits[idx]),
#                     int(counts.difference[idx]),
#                 ]
#             )


# def _write_summary_csv(out_path: Path, eval_h5: Path, counts: EventCounts, fit: LinearFit | None) -> None:
#     init_stats = _stats(counts.num_initialised_queries)
#     truth_stats = _stats(counts.num_reconstructable_particles)
#     hits_stats = _stats(counts.num_pre_filter_hits)
#     diff_stats = _stats(counts.difference)

#     with out_path.open("w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["metric", "value"])
#         writer.writerow(["eval_h5", str(eval_h5)])
#         writer.writerow(["num_events", len(counts.sample_ids)])
#         writer.writerow([])
#         writer.writerow(["initialised_queries_min", init_stats["min"]])
#         writer.writerow(["initialised_queries_mean", init_stats["mean"]])
#         writer.writerow(["initialised_queries_std", init_stats["std"]])
#         writer.writerow(["initialised_queries_max", init_stats["max"]])
#         writer.writerow([])
#         writer.writerow(["reconstructable_particles_min", truth_stats["min"]])
#         writer.writerow(["reconstructable_particles_mean", truth_stats["mean"]])
#         writer.writerow(["reconstructable_particles_std", truth_stats["std"]])
#         writer.writerow(["reconstructable_particles_max", truth_stats["max"]])
#         writer.writerow([])
#         writer.writerow(["pre_filter_hits_min", hits_stats["min"]])
#         writer.writerow(["pre_filter_hits_mean", hits_stats["mean"]])
#         writer.writerow(["pre_filter_hits_std", hits_stats["std"]])
#         writer.writerow(["pre_filter_hits_max", hits_stats["max"]])
#         writer.writerow([])
#         writer.writerow(["difference_min", diff_stats["min"]])
#         writer.writerow(["difference_mean", diff_stats["mean"]])
#         writer.writerow(["difference_std", diff_stats["std"]])
#         writer.writerow(["difference_max", diff_stats["max"]])
#         writer.writerow(["linear_fit_intercept", "" if fit is None else fit.intercept])
#         writer.writerow(["linear_fit_slope", "" if fit is None else fit.slope])
#         writer.writerow(["linear_fit_r2", "" if fit is None else fit.r2])


# def _print_summary(counts: EventCounts) -> None:
#     init_stats = _stats(counts.num_initialised_queries)
#     truth_stats = _stats(counts.num_reconstructable_particles)
#     hits_stats = _stats(counts.num_pre_filter_hits)
#     diff_stats = _stats(counts.difference)

#     print(
#         "initialised queries per event: "
#         f"min={int(init_stats['min'])}, max={int(init_stats['max'])}, "
#         f"mean={init_stats['mean']:.3f}, std={init_stats['std']:.3f}"
#     )
#     print(
#         "Reconstructable truth particles per event: "
#         f"min={int(truth_stats['min'])}, max={int(truth_stats['max'])}, "
#         f"mean={truth_stats['mean']:.3f}, std={truth_stats['std']:.3f}"
#     )
#     print(
#         "Pre-filter hits per event: "
#         f"min={int(hits_stats['min'])}, max={int(hits_stats['max'])}, "
#         f"mean={hits_stats['mean']:.3f}, std={hits_stats['std']:.3f}"
#     )
#     print(
#         "initialised queries minus reconstructable particles: "
#         f"min={int(diff_stats['min'])}, max={int(diff_stats['max'])}, "
#         f"mean={diff_stats['mean']:.3f}, std={diff_stats['std']:.3f}"
#     )


# def main() -> None:
#     os.environ.setdefault("MPLBACKEND", "Agg")
#     args = parse_args()

#     if not args.eval_h5.exists():
#         raise FileNotFoundError(f"Eval file does not exist: {args.eval_h5}")

#     output_dir = args.output_dir if args.output_dir is not None else args.eval_h5.parent / "query_init_event_plots"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     output_base = output_dir / args.output_stem

#     counts = _collect_event_counts(eval_h5=args.eval_h5, max_events=args.max_events)
#     fit = _plot_figure(counts=counts, add_linear_fit=args.add_linear_fit, output_base=output_base)
#     _write_per_event_csv(output_base.with_name(output_base.name + "_per_event.csv"), counts=counts)
#     _write_summary_csv(output_base.with_name(output_base.name + "_summary.csv"), eval_h5=args.eval_h5, counts=counts, fit=fit)
#     _print_summary(counts)

#     print(f"Saved figure to {output_base.with_suffix('.pdf')}")
#     print(f"Saved figure to {output_base.with_suffix('.png')}")
#     print(f"Saved per-event table to {output_base.with_name(output_base.name + '_per_event.csv')}")
#     print(f"Saved summary to {output_base.with_name(output_base.name + '_summary.csv')}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""Plot initialised-query counts against reconstructable truth particles per event."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class EventCounts:
    sample_ids: list[str]
    num_initialised_queries: np.ndarray
    num_reconstructable_particles: np.ndarray
    num_pre_filter_hits: np.ndarray

    @property
    def difference(self) -> np.ndarray:
        return self.num_initialised_queries - self.num_reconstructable_particles


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a TrackML tracking eval HDF5 file and plot initialised queries per event against reconstructable truth particles per event."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-h5",
        type=Path,
        required=True,
        help="Path to a tracking eval HDF5. Supports both the old `targets/...` layout and the `PredictionWriter` paper-format layout.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum number of events to process. Use -1 for all available events.",
    )
    parser.add_argument(
        "--add-linear-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay a least-squares linear fit on the scatter plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the figure and CSV summaries are written. Defaults to <eval_h5_parent>/query_init_event_plots.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="initialised_queries_vs_reconstructable_particles",
        help="Filename stem used for output files.",
    )
    parser.add_argument(
        "--eta-cut",
        type=float,
        default=4.0,
        help="Absolute eta cut used when reconstructable counts must be derived from a paper-format file.",
    )
    parser.add_argument(
        "--pt-cut",
        type=float,
        default=1.0,
        help="pT cut used when reconstructable counts must be derived from a paper-format file.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum number of truth hits used when reconstructable counts must be derived from a paper-format file.",
    )
    return parser.parse_args()


def _get_pyplot():
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate initialised-query plots. Install it in the active environment and rerun."
        ) from exc

    mpl.use("Agg", force=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfc",
        "savefig.facecolor": "white",
    })
    import matplotlib.pyplot as plt

    return plt


def _sorted_event_keys(h5f: h5py.File) -> list[str]:
    def _key(key: str) -> tuple[int, int | str]:
        if key.isdigit():
            return (0, int(key))
        if key.startswith("event_") and key.removeprefix("event_").isdigit():
            return (1, int(key.removeprefix("event_")))
        return (2, key)

    return sorted(h5f.keys(), key=_key)


def _read_array(group: h5py.Group, path: str, required: bool = True) -> np.ndarray | None:
    try:
        array = np.asarray(group[path])
    except KeyError:
        if required:
            raise
        return None
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _read_scalar_int(group: h5py.Group, path: str, required: bool = True) -> int | None:
    array = _read_array(group, path, required=required)
    if array is None:
        return None
    return int(np.asarray(array).reshape(-1)[0])


def _sample_id_for_group(group: h5py.Group, fallback: str) -> str:
    sample_id = group.attrs.get("sample_id")
    if sample_id is None:
        return fallback
    if isinstance(sample_id, bytes):
        return sample_id.decode()
    if isinstance(sample_id, np.ndarray):
        sample_id = sample_id.item()
    return str(sample_id)


def _is_paper_format_group(group: h5py.Group) -> bool:
    return all(name in group for name in ("truth", "hits", "parts", "preds"))


def _collect_old_format_counts(group: h5py.Group) -> tuple[int, int, int]:
    init_queries = _read_scalar_int(group, "targets/num_initialised_queries", required=False)
    if init_queries is None:
        query_mask = _read_array(group, "targets/query_mask", required=False)
        if query_mask is None:
            track_valid = _read_array(group, "preds/final/track_valid/track_valid", required=False)
            if track_valid is None:
                raise KeyError("Eval file is missing both targets/num_initialised_queries and fallback query-count datasets.")
            init_queries = int(np.asarray(track_valid).astype(bool).shape[-1])
        else:
            init_queries = int(np.asarray(query_mask).astype(bool).sum())

    reconstructable = _read_scalar_int(group, "targets/num_reconstructable_particles", required=False)
    if reconstructable is None:
        particle_valid = _read_array(group, "targets/particle_valid")
        reconstructable = int(np.asarray(particle_valid).astype(bool).sum())

    phv = _read_array(group, "targets/particle_hit_valid", required=False)
    if phv is not None:
        pre_hits = int(np.asarray(phv).shape[-1])
    else:
        pre_hits = 0
        if "targets" in group:
            for ds_name in group["targets"]:
                if ds_name.endswith("_valid") and ds_name not in (
                    "particle_valid",
                    "query_mask",
                    "num_initialised_queries",
                    "num_reconstructable_particles",
                ):
                    arr = _read_array(group, f"targets/{ds_name}", required=False)
                    if arr is not None and np.asarray(arr).ndim == 1:
                        pre_hits += int(np.asarray(arr).shape[0])

    return init_queries, reconstructable, pre_hits


def _collect_paper_format_counts(
    group: h5py.Group,
    *,
    eta_cut: float,
    pt_cut: float,
    min_hits: int,
) -> tuple[int, int, int]:
    init_queries = _read_scalar_int(group, "preds/num_initialised_queries", required=False)
    if init_queries is None:
        query_mask = _read_array(group, "preds/query_mask", required=False)
        if query_mask is not None:
            init_queries = int(np.asarray(query_mask).astype(bool).sum())
    if init_queries is None and "masks" in group["preds"]:
        init_queries = int(np.asarray(group["preds"]["masks"]).shape[0])
    elif init_queries is None and "class_preds" in group["preds"]:
        init_queries = int(np.asarray(group["preds"]["class_preds"]).shape[0])
    elif init_queries is None and "track_valid_prob" in group["preds"]:
        init_queries = int(np.asarray(group["preds"]["track_valid_prob"]).shape[0])
    if init_queries is None:
        raise KeyError("Paper-format eval file is missing `preds/masks`, `preds/class_preds`, and `preds/track_valid_prob`.")

    particle_ids = np.asarray(group["parts"]["pids"], dtype=np.int64)
    particle_pts = np.asarray(group["parts"]["pts"], dtype=np.float32)
    particle_etas = np.asarray(group["parts"]["etas"], dtype=np.float32)
    particle_n_hits = np.asarray(group["parts"]["n_hits"], dtype=np.int64)
    valid_truth = particle_ids > 0
    reconstructable = int((valid_truth & (particle_n_hits >= min_hits) & (np.abs(particle_etas) < eta_cut) & (particle_pts > pt_cut)).sum())

    # Paper-format files keep per-particle pre-filter hit counts but not the full
    # pre-filter hit table, so approximate the event hit count from that truth summary.
    pre_hits = int(particle_n_hits[valid_truth].sum())
    return init_queries, reconstructable, pre_hits


def _collect_event_counts(eval_h5: Path, max_events: int, eta_cut: float, pt_cut: float, min_hits: int) -> EventCounts:
    with h5py.File(eval_h5, "r") as h5f:
        event_keys = _sorted_event_keys(h5f)
        if max_events > 0:
            event_keys = event_keys[:max_events]
        if not event_keys:
            raise RuntimeError(f"No events found in eval file: {eval_h5}")

        n_events = len(event_keys)
        sample_ids: list[str] = []
        num_initialised_queries = np.zeros(n_events, dtype=np.int32)
        num_reconstructable_particles = np.zeros(n_events, dtype=np.int32)
        num_pre_filter_hits = np.zeros(n_events, dtype=np.int32)

        for i, key in enumerate(event_keys):
            group = h5f[key]
            sample_ids.append(_sample_id_for_group(group, fallback=key))

            if _is_paper_format_group(group):
                init_queries, reconstructable, pre_hits = _collect_paper_format_counts(
                    group,
                    eta_cut=eta_cut,
                    pt_cut=pt_cut,
                    min_hits=min_hits,
                )
            else:
                init_queries, reconstructable, pre_hits = _collect_old_format_counts(group)

            num_initialised_queries[i] = init_queries
            num_reconstructable_particles[i] = reconstructable
            num_pre_filter_hits[i] = pre_hits

            if (i + 1) % 50 == 0 or (i + 1) == n_events:
                print(f"Processed {i + 1}/{n_events} events")

    return EventCounts(
        sample_ids=sample_ids,
        num_initialised_queries=num_initialised_queries,
        num_reconstructable_particles=num_reconstructable_particles,
        num_pre_filter_hits=num_pre_filter_hits,
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> LinearFit | None:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 2 or np.unique(x).size < 2:
        return None

    slope, intercept = np.polyfit(x, y, deg=1)
    pred = intercept + slope * x
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LinearFit(slope=float(slope), intercept=float(intercept), r2=float(r2))


def _hist_bins(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        center = float(values[0])
        return np.array([center - 0.5, center + 0.5], dtype=float)
    if np.all(values == values.astype(np.int64)) and (values.max() - values.min()) <= 200:
        return np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
    n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
    return np.linspace(values.min(), values.max(), n_bins + 1)


def _plot_figure(
    counts: EventCounts,
    add_linear_fit: bool,
    output_base: Path,
) -> LinearFit | None:
    plt = _get_pyplot()
    fit = _linear_fit(counts.num_reconstructable_particles, counts.num_initialised_queries) if add_linear_fit else None
    hits_fit = _linear_fit(counts.num_pre_filter_hits, counts.num_reconstructable_particles) if add_linear_fit else None
    diff = counts.difference

    x = counts.num_reconstructable_particles.astype(np.float64)
    y = counts.num_initialised_queries.astype(np.float64)
    fig, scatter_ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    scatter_ax.scatter(
        x,
        y,
        s=20.0,
        marker=".",
        alpha=0.65,
        color="#1f77b4",
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    if fit is not None:
        x_line = np.linspace(float(x.min()), float(x.max()) * 1.02, 256)
        scatter_ax.plot(
            x_line,
            fit.intercept + fit.slope * x_line,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            zorder=3,
        )
        scatter_ax.text(
            0.06,
            0.94,
            f"Slope = {fit.slope:.4f}\n$R^2$ = {fit.r2:.2f}",
            transform=scatter_ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )

    scatter_ax.set_xlabel("Reconstructable truth particles per event")
    scatter_ax.set_ylabel("initialised queries per event")
    scatter_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    scatter_ax.set_axisbelow(True)
    fig.savefig(output_base.with_name(output_base.name + "_queries_vs_reconstructable").with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_name(output_base.name + "_queries_vs_reconstructable").with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    diff_stats = _stats(diff)
    bins = _hist_bins(diff)
    fig, hist_ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    hist_ax.hist(diff, bins=bins, alpha=0.85, color="#d62728", edgecolor="black", linewidth=0.35)
    hist_ax.axvline(diff_stats["mean"], color="#444444", linestyle="--", linewidth=1.2)
    hist_ax.set_xlabel("initialised queries - reconstructable particles")
    hist_ax.set_ylabel("Events")
    hist_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    hist_ax.set_axisbelow(True)
    hist_ax.text(
        0.97,
        0.94,
        f"Mean = {diff_stats['mean']:.2f}\nStd = {diff_stats['std']:.2f}",
        transform=hist_ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    fig.savefig(output_base.with_name(output_base.name + "_difference_hist").with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_name(output_base.name + "_difference_hist").with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    hx = counts.num_pre_filter_hits.astype(np.float64)
    hy = counts.num_reconstructable_particles.astype(np.float64)
    fig, hits_ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    hits_ax.scatter(
        hx,
        hy,
        s=20.0,
        marker=".",
        alpha=0.65,
        color="#2ca02c",
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    if hits_fit is not None:
        hx_line = np.linspace(float(hx.min()), float(hx.max()) * 1.02, 256)
        hits_ax.plot(
            hx_line,
            hits_fit.intercept + hits_fit.slope * hx_line,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            zorder=3,
        )
        hits_ax.text(
            0.06,
            0.94,
            f"Slope = {hits_fit.slope:.4f}\n$R^2$ = {hits_fit.r2:.2f}",
            transform=hits_ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
    hits_ax.set_xlabel("Pre-filter hits per event")
    hits_ax.set_ylabel("Reconstructable truth particles per event")
    hits_ax.grid(alpha=0.10, linestyle="--", linewidth=0.45)
    hits_ax.set_axisbelow(True)
    fig.savefig(output_base.with_name(output_base.name + "_hits_vs_reconstructable").with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_name(output_base.name + "_hits_vs_reconstructable").with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fit


def _write_per_event_csv(out_path: Path, counts: EventCounts) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_index",
            "sample_id",
            "num_initialised_queries",
            "num_reconstructable_particles",
            "num_pre_filter_hits",
            "initialised_queries_minus_reconstructable_particles",
        ])
        for idx, sample_id in enumerate(counts.sample_ids):
            writer.writerow([
                idx,
                sample_id,
                int(counts.num_initialised_queries[idx]),
                int(counts.num_reconstructable_particles[idx]),
                int(counts.num_pre_filter_hits[idx]),
                int(counts.difference[idx]),
            ])


def _write_summary_csv(out_path: Path, eval_h5: Path, counts: EventCounts, fit: LinearFit | None) -> None:
    init_stats = _stats(counts.num_initialised_queries)
    truth_stats = _stats(counts.num_reconstructable_particles)
    hits_stats = _stats(counts.num_pre_filter_hits)
    diff_stats = _stats(counts.difference)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["eval_h5", str(eval_h5)])
        writer.writerow(["num_events", len(counts.sample_ids)])
        writer.writerow([])
        writer.writerow(["initialised_queries_min", init_stats["min"]])
        writer.writerow(["initialised_queries_mean", init_stats["mean"]])
        writer.writerow(["initialised_queries_std", init_stats["std"]])
        writer.writerow(["initialised_queries_max", init_stats["max"]])
        writer.writerow([])
        writer.writerow(["reconstructable_particles_min", truth_stats["min"]])
        writer.writerow(["reconstructable_particles_mean", truth_stats["mean"]])
        writer.writerow(["reconstructable_particles_std", truth_stats["std"]])
        writer.writerow(["reconstructable_particles_max", truth_stats["max"]])
        writer.writerow([])
        writer.writerow(["pre_filter_hits_min", hits_stats["min"]])
        writer.writerow(["pre_filter_hits_mean", hits_stats["mean"]])
        writer.writerow(["pre_filter_hits_std", hits_stats["std"]])
        writer.writerow(["pre_filter_hits_max", hits_stats["max"]])
        writer.writerow([])
        writer.writerow(["difference_min", diff_stats["min"]])
        writer.writerow(["difference_mean", diff_stats["mean"]])
        writer.writerow(["difference_std", diff_stats["std"]])
        writer.writerow(["difference_max", diff_stats["max"]])
        writer.writerow(["linear_fit_intercept", "" if fit is None else fit.intercept])
        writer.writerow(["linear_fit_slope", "" if fit is None else fit.slope])
        writer.writerow(["linear_fit_r2", "" if fit is None else fit.r2])


def _print_summary(counts: EventCounts) -> None:
    init_stats = _stats(counts.num_initialised_queries)
    truth_stats = _stats(counts.num_reconstructable_particles)
    hits_stats = _stats(counts.num_pre_filter_hits)
    diff_stats = _stats(counts.difference)

    print(
        "initialised queries per event: "
        f"min={int(init_stats['min'])}, max={int(init_stats['max'])}, "
        f"mean={init_stats['mean']:.3f}, std={init_stats['std']:.3f}"
    )
    print(
        "Reconstructable truth particles per event: "
        f"min={int(truth_stats['min'])}, max={int(truth_stats['max'])}, "
        f"mean={truth_stats['mean']:.3f}, std={truth_stats['std']:.3f}"
    )
    print(
        "Pre-filter hits per event: "
        f"min={int(hits_stats['min'])}, max={int(hits_stats['max'])}, "
        f"mean={hits_stats['mean']:.3f}, std={hits_stats['std']:.3f}"
    )
    print(
        "initialised queries minus reconstructable particles: "
        f"min={int(diff_stats['min'])}, max={int(diff_stats['max'])}, "
        f"mean={diff_stats['mean']:.3f}, std={diff_stats['std']:.3f}"
    )


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    if not args.eval_h5.exists():
        raise FileNotFoundError(f"Eval file does not exist: {args.eval_h5}")

    output_dir = args.output_dir if args.output_dir is not None else args.eval_h5.parent / "query_init_event_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / args.output_stem

    counts = _collect_event_counts(
        eval_h5=args.eval_h5,
        max_events=args.max_events,
        eta_cut=args.eta_cut,
        pt_cut=args.pt_cut,
        min_hits=args.min_hits,
    )
    fit = _plot_figure(counts=counts, add_linear_fit=args.add_linear_fit, output_base=output_base)
    _write_per_event_csv(output_base.with_name(output_base.name + "_per_event.csv"), counts=counts)
    _write_summary_csv(output_base.with_name(output_base.name + "_summary.csv"), eval_h5=args.eval_h5, counts=counts, fit=fit)
    _print_summary(counts)

    print(f"Saved figure to {output_base.with_name(output_base.name + '_queries_vs_reconstructable').with_suffix('.pdf')}")
    print(f"Saved figure to {output_base.with_name(output_base.name + '_queries_vs_reconstructable').with_suffix('.png')}")
    print(f"Saved figure to {output_base.with_name(output_base.name + '_difference_hist').with_suffix('.pdf')}")
    print(f"Saved figure to {output_base.with_name(output_base.name + '_difference_hist').with_suffix('.png')}")
    print(f"Saved figure to {output_base.with_name(output_base.name + '_hits_vs_reconstructable').with_suffix('.pdf')}")
    print(f"Saved figure to {output_base.with_name(output_base.name + '_hits_vs_reconstructable').with_suffix('.png')}")
    print(f"Saved per-event table to {output_base.with_name(output_base.name + '_per_event.csv')}")
    print(f"Saved summary to {output_base.with_name(output_base.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
