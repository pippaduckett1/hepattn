#!/usr/bin/env python3
"""Plot hit-coordinate histograms comparing origin- and vertex-based definitions.

The figure contains two overlaid 1D histograms:

1. Hit phi, computed once with respect to the detector origin (0, 0, 0) and
   once with respect to the truth production vertex (vx, vy, vz).
2. Hit eta, computed with the same two coordinate definitions.

The script uses the TrackML dataset loader and tracking config so volume
selection, reconstructability cuts, and optional hit-filter selection follow the
existing TrackML setup.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CONFIG = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml")
DEFAULT_OUTPUT_DIR = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/PAPERPLOTTING/")
DEFAULT_OUTPUT_STEM = "phi_eta_origin_vs_vertex_hist"
HIT_COORDINATE_SCALE = 0.01
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create overlaid hit histograms in phi and eta, comparing coordinates "
            "computed from (0,0,0) against coordinates computed from the truth "
            "primary vertex."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to a TrackML YAML config.")
    parser.add_argument("--split", choices=VALID_SPLITS, default="test", help="Dataset split to analyze.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum number of events to process. Use -1 to use all selected events.",
    )
    parser.add_argument(
        "--use-hit-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the split-specific hit-filter evaluation file from the config while loading events.",
    )
    parser.add_argument(
        "--volume-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional detector-volume override. Omit to use the config selection. "
            "Use '--volume-ids 8' for a barrel-only figure."
        ),
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=-1,
        help="Maximum number of valid hits kept after loading. Use -1 to keep all hits.",
    )
    parser.add_argument("--phi-bins", type=int, default=72, help="Number of bins for the phi histogram.")
    parser.add_argument("--eta-bins", type=int, default=80, help="Number of bins for the eta histogram.")
    parser.add_argument(
        "--eta-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("ETA_MIN", "ETA_MAX"),
        help="Explicit eta histogram range. Omit to infer it from the selected hits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figure and histogram CSV files are written.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem used for output files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for optional hit downsampling.")
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
            "matplotlib is required to generate the phi/eta histogram figure. "
            "Install it in the active environment and rerun."
        ) from exc

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.8,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def _build_dataset(cfg: dict, split: str, use_hit_eval: bool, volume_ids: list[int] | None):
    from data import TrackMLDataset

    data_cfg = cfg["data"]
    split_dir = Path(data_cfg[f"{split}_dir"])
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    split_num = int(data_cfg.get(f"num_{split}", -1))
    num_events = split_num if split_num > 0 else -1

    hit_eval_value = data_cfg.get(f"hit_eval_{split}")
    if use_hit_eval and not hit_eval_value:
        raise ValueError(f"Config does not define data.hit_eval_{split} for split '{split}'.")

    hit_eval_path = Path(hit_eval_value) if use_hit_eval and hit_eval_value else None
    if hit_eval_path is not None and not hit_eval_path.exists():
        raise FileNotFoundError(f"Configured hit-eval file does not exist for split '{split}': {hit_eval_path}")

    dataset = TrackMLDataset(
        dirpath=str(split_dir),
        inputs=data_cfg["inputs"],
        targets=data_cfg["targets"],
        num_events=num_events,
        hit_volume_ids=volume_ids if volume_ids is not None else data_cfg.get("hit_volume_ids"),
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


def _downsample_arrays(arrays: dict[str, np.ndarray], max_hits: int, seed: int) -> tuple[dict[str, np.ndarray], int]:
    n_hits = len(next(iter(arrays.values())))
    if max_hits < 0 or n_hits <= max_hits:
        return arrays, n_hits

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_hits, size=max_hits, replace=False))
    return {key: value[keep] for key, value in arrays.items()}, n_hits


def _collect_valid_hits(
    dataset,
    max_events: int,
    max_hits: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], int]:
    event_limit = len(dataset) if max_events < 0 else min(max_events, len(dataset))
    if event_limit <= 0:
        raise RuntimeError("No events selected.")

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    r_chunks: list[np.ndarray] = []
    vx_chunks: list[np.ndarray] = []
    vy_chunks: list[np.ndarray] = []
    vz_chunks: list[np.ndarray] = []

    for idx in range(event_limit):
        hits, particles = dataset.load_event(idx)
        valid_hits = hits[hits["on_valid_particle"]]
        if len(valid_hits) == 0:
            continue

        required_particle_fields = {"particle_id", "vx", "vy", "vz"}
        missing = required_particle_fields.difference(particles.columns)
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise KeyError(
                "The particle parquet files do not contain the truth-vertex fields "
                f"required for this plot: {missing_fields}"
            )

        merged = valid_hits.merge(
            particles[["particle_id", "vx", "vy", "vz"]],
            on="particle_id",
            how="inner",
            validate="many_to_one",
        )
        if len(merged) == 0:
            continue

        x_chunks.append(merged["x"].to_numpy(dtype=np.float64))
        y_chunks.append(merged["y"].to_numpy(dtype=np.float64))
        z_chunks.append(merged["z"].to_numpy(dtype=np.float64))
        r_chunks.append(merged["r"].to_numpy(dtype=np.float64))
        vx_chunks.append(merged["vx"].to_numpy(dtype=np.float64) * HIT_COORDINATE_SCALE)
        vy_chunks.append(merged["vy"].to_numpy(dtype=np.float64) * HIT_COORDINATE_SCALE)
        vz_chunks.append(merged["vz"].to_numpy(dtype=np.float64) * HIT_COORDINATE_SCALE)

        if (idx + 1) % 25 == 0 or (idx + 1) == event_limit:
            print(f"Processed {idx + 1}/{event_limit} events")

    if not x_chunks:
        raise RuntimeError("No valid hits were found after applying the dataset selections.")

    arrays = {
        "x": np.concatenate(x_chunks),
        "y": np.concatenate(y_chunks),
        "z": np.concatenate(z_chunks),
        "r": np.concatenate(r_chunks),
        "vx": np.concatenate(vx_chunks),
        "vy": np.concatenate(vy_chunks),
        "vz": np.concatenate(vz_chunks),
    }
    return _downsample_arrays(arrays=arrays, max_hits=max_hits, seed=seed)


def _compute_coordinate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    dx = arrays["x"] - arrays["vx"]
    dy = arrays["y"] - arrays["vy"]
    dz = arrays["z"] - arrays["vz"]
    rho_origin = arrays["r"]
    rho_vertex = np.sqrt(dx**2 + dy**2)

    valid = np.isfinite(rho_origin) & np.isfinite(rho_vertex) & (rho_origin > 0.0) & (rho_vertex > 0.0)
    if not np.any(valid):
        raise RuntimeError("No finite hit coordinates remain after constructing origin- and vertex-based coordinates.")

    phi_origin = np.arctan2(arrays["y"][valid], arrays["x"][valid])
    phi_vertex = np.arctan2(dy[valid], dx[valid])
    eta_origin = np.arcsinh(arrays["z"][valid] / rho_origin[valid])
    eta_vertex = np.arcsinh(dz[valid] / rho_vertex[valid])

    return {
        "phi_origin": phi_origin,
        "phi_vertex": phi_vertex,
        "eta_origin": eta_origin,
        "eta_vertex": eta_vertex,
    }


def _infer_eta_range(eta_origin: np.ndarray, eta_vertex: np.ndarray) -> tuple[float, float]:
    all_eta = np.concatenate([eta_origin, eta_vertex])
    finite_eta = all_eta[np.isfinite(all_eta)]
    if finite_eta.size == 0:
        raise RuntimeError("No finite eta values remain for histogramming.")

    eta_min = float(np.min(finite_eta))
    eta_max = float(np.max(finite_eta))
    if np.isclose(eta_min, eta_max):
        eta_min -= 0.5
        eta_max += 0.5
    return eta_min, eta_max


def _build_hist_rows(
    coordinate_name: str,
    values_origin: np.ndarray,
    values_vertex: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    counts_origin, _ = np.histogram(values_origin, bins=edges)
    counts_vertex, _ = np.histogram(values_vertex, bins=edges)

    rows: list[dict[str, float]] = []
    for low, high, count_origin, count_vertex in zip(
        edges[:-1],
        edges[1:],
        counts_origin,
        counts_vertex,
        strict=True,
    ):
        rows.append(
            {
                "coordinate": coordinate_name,
                "bin_low": float(low),
                "bin_high": float(high),
                "count_origin": int(count_origin),
                "count_truth_vertex": int(count_vertex),
            }
        )
    return counts_origin, counts_vertex, rows


def _write_hist_csv(out_path: Path, rows: list[dict[str, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["coordinate", "bin_low", "bin_high", "count_origin", "count_truth_vertex"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / float(total)


def _plot_histograms(
    output_base: Path,
    phi_edges: np.ndarray,
    phi_counts_origin: np.ndarray,
    phi_counts_vertex: np.ndarray,
    eta_edges: np.ndarray,
    eta_counts_origin: np.ndarray,
    eta_counts_vertex: np.ndarray,
    split: str,
    event_count: int,
    hits_used: int,
    hits_available: int,
):
    plt = _get_pyplot()
    from matplotlib import ticker

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 5.6),
        sharex="col",
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.05, "wspace": 0.16},
    )
    ax_phi, ax_eta = axes[0]
    ax_phi_resid, ax_eta_resid = axes[1]

    color_origin = "#2F5D8C"
    color_vertex = "#C94C4C"
    residual_fill = "#B8B8B8"
    phi_density_origin = _normalize_counts(phi_counts_origin)
    phi_density_vertex = _normalize_counts(phi_counts_vertex)
    eta_density_origin = _normalize_counts(eta_counts_origin)
    eta_density_vertex = _normalize_counts(eta_counts_vertex)

    phi_origin_artist = ax_phi.stairs(
        phi_density_origin,
        phi_edges,
        color=color_origin,
        linewidth=2.0,
        label=r"wrt $(0,0,0)$",
    )
    ax_phi.stairs(phi_density_origin, phi_edges, fill=True, baseline=0.0, color=color_origin, alpha=0.08, linewidth=0.0)
    phi_vertex_artist = ax_phi.stairs(
        phi_density_vertex,
        phi_edges,
        color=color_vertex,
        linewidth=2.0,
        label=r"wrt truth vertex",
    )
    ax_phi.stairs(phi_density_vertex, phi_edges, fill=True, baseline=0.0, color=color_vertex, alpha=0.08, linewidth=0.0)
    ax_phi.set_ylabel("Hits per bin [%]")
    ax_phi.grid(axis="both", alpha=0.55, linestyle="--")

    ax_eta.stairs(eta_density_origin, eta_edges, color=color_origin, linewidth=2.0, label=r"wrt $(0,0,0)$")
    ax_eta.stairs(eta_density_origin, eta_edges, fill=True, baseline=0.0, color=color_origin, alpha=0.08, linewidth=0.0)
    ax_eta.stairs(eta_density_vertex, eta_edges, color=color_vertex, linewidth=2.0, label=r"wrt truth vertex")
    ax_eta.stairs(eta_density_vertex, eta_edges, fill=True, baseline=0.0, color=color_vertex, alpha=0.08, linewidth=0.0)
    ax_eta.grid(axis="both", alpha=0.55, linestyle="--")

    phi_delta = phi_density_vertex - phi_density_origin
    eta_delta = eta_density_vertex - eta_density_origin

    ax_phi_resid.axhline(0.0, color="#555555", linewidth=0.9)
    ax_phi_resid.stairs(phi_delta, phi_edges, color="#555555", linewidth=1.4)
    ax_phi_resid.stairs(phi_delta, phi_edges, fill=True, baseline=0.0, color=residual_fill, alpha=0.35, linewidth=0.0)
    ax_phi_resid.set_xlabel(r"$\phi$")
    ax_phi_resid.set_ylabel(r"$\Delta$ [%]")
    ax_phi_resid.grid(axis="y", alpha=0.4, linestyle="--")

    ax_eta_resid.axhline(0.0, color="#555555", linewidth=0.9)
    ax_eta_resid.stairs(eta_delta, eta_edges, color="#555555", linewidth=1.4)
    ax_eta_resid.stairs(eta_delta, eta_edges, fill=True, baseline=0.0, color=residual_fill, alpha=0.35, linewidth=0.0)
    ax_eta_resid.set_xlabel(r"$\eta$")
    ax_eta_resid.grid(axis="y", alpha=0.4, linestyle="--")

    shared_top_lim = 1.05 * max(
        float(np.max(phi_density_origin)),
        float(np.max(phi_density_vertex)),
        float(np.max(eta_density_origin)),
        float(np.max(eta_density_vertex)),
    )
    ax_phi.set_ylim(0.0, shared_top_lim)
    ax_eta.set_ylim(0.0, shared_top_lim)

    shared_resid_lim = 1.05 * max(
        float(np.max(np.abs(phi_delta))),
        float(np.max(np.abs(eta_delta))),
    )
    if shared_resid_lim == 0.0:
        shared_resid_lim = 1.0
    ax_phi_resid.set_ylim(-shared_resid_lim, shared_resid_lim)
    ax_eta_resid.set_ylim(-shared_resid_lim, shared_resid_lim)
    ax_eta_resid.set_ylabel("")

    percent_formatter = ticker.PercentFormatter(xmax=1.0, decimals=1)
    resid_formatter = ticker.PercentFormatter(xmax=1.0, decimals=2)
    for ax in (ax_phi, ax_eta):
        ax.yaxis.set_major_formatter(percent_formatter)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.tick_params(direction="out", length=3.5)
    for ax in (ax_phi_resid, ax_eta_resid):
        ax.yaxis.set_major_formatter(resid_formatter)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.tick_params(direction="out", length=3.0)

    ax_phi.set_xlim(phi_edges[0], phi_edges[-1])
    ax_eta.set_xlim(eta_edges[0], eta_edges[-1])
    ax_phi_resid.set_xlim(phi_edges[0], phi_edges[-1])
    ax_eta_resid.set_xlim(eta_edges[0], eta_edges[-1])

    ax_eta.tick_params(labelleft=False)
    ax_eta_resid.tick_params(labelleft=False)

    for ax in (ax_phi, ax_eta, ax_phi_resid, ax_eta_resid):
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    ax_phi.legend(
        handles=[phi_origin_artist, phi_vertex_artist],
        labels=[r"wrt $(0,0,0)$", r"wrt truth vertex"],
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    fig.subplots_adjust(top=0.97, bottom=0.13, left=0.09, right=0.98)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = parse_args()

    if args.phi_bins < 1:
        raise ValueError(f"--phi-bins must be >= 1, got {args.phi_bins}.")
    if args.eta_bins < 1:
        raise ValueError(f"--eta-bins must be >= 1, got {args.eta_bins}.")
    if args.eta_range is not None and args.eta_range[0] >= args.eta_range[1]:
        raise ValueError(f"--eta-range must satisfy ETA_MIN < ETA_MAX, got {args.eta_range}.")

    cfg = _read_yaml(args.config)
    dataset, split_dir, hit_eval_path = _build_dataset(
        cfg=cfg,
        split=args.split,
        use_hit_eval=args.use_hit_eval,
        volume_ids=args.volume_ids,
    )

    event_count = len(dataset) if args.max_events < 0 else min(args.max_events, len(dataset))
    print(f"[{args.split}] loading up to {event_count} events from {split_dir}")
    if args.use_hit_eval:
        print(f"[{args.split}] applying hit-eval file {hit_eval_path}")

    arrays, hits_available = _collect_valid_hits(
        dataset=dataset,
        max_events=args.max_events,
        max_hits=args.max_hits,
        seed=args.seed,
    )
    coordinates = _compute_coordinate_arrays(arrays)
    hits_used = len(coordinates["phi_origin"])

    phi_edges = np.linspace(-np.pi, np.pi, args.phi_bins + 1)
    eta_range = tuple(args.eta_range) if args.eta_range is not None else _infer_eta_range(
        coordinates["eta_origin"],
        coordinates["eta_vertex"],
    )
    eta_edges = np.linspace(eta_range[0], eta_range[1], args.eta_bins + 1)

    phi_counts_origin, phi_counts_vertex, phi_rows = _build_hist_rows(
        coordinate_name="phi",
        values_origin=coordinates["phi_origin"],
        values_vertex=coordinates["phi_vertex"],
        edges=phi_edges,
    )
    eta_counts_origin, eta_counts_vertex, eta_rows = _build_hist_rows(
        coordinate_name="eta",
        values_origin=coordinates["eta_origin"],
        values_vertex=coordinates["eta_vertex"],
        edges=eta_edges,
    )

    output_base = args.output_dir / args.output_stem
    _plot_histograms(
        output_base=output_base,
        phi_edges=phi_edges,
        phi_counts_origin=phi_counts_origin,
        phi_counts_vertex=phi_counts_vertex,
        eta_edges=eta_edges,
        eta_counts_origin=eta_counts_origin,
        eta_counts_vertex=eta_counts_vertex,
        split=args.split,
        event_count=event_count,
        hits_used=hits_used,
        hits_available=hits_available,
    )
    _write_hist_csv(output_base.with_name(output_base.name + "_phi_hist.csv"), phi_rows)
    _write_hist_csv(output_base.with_name(output_base.name + "_eta_hist.csv"), eta_rows)

    print(f"Saved figure to {output_base.with_suffix('.png')}")
    print(f"Saved figure to {output_base.with_suffix('.pdf')}")
    print(f"Saved phi histogram to {output_base.with_name(output_base.name + '_phi_hist.csv')}")
    print(f"Saved eta histogram to {output_base.with_name(output_base.name + '_eta_hist.csv')}")


if __name__ == "__main__":
    main()
