"""Assess per-event hit and particle counts from TrackML data.

This script uses TrackMLDataset.load_event() so event-level counts reflect
TrackML preprocessing cuts (volume selection, particle cuts, optional hit_eval).
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hepattn.experiments.trackml.data import TrackMLDataset  # noqa: E402

mpl.use("Agg", force=True)


@dataclass
class Settings:
    config_path: Path = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600.yaml")
    split: str = "train"  # train, val, test
    max_events: int = -1  # -1 uses all available from config/directory
    use_hit_eval: bool = True
    output_dir: Path = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots_600/")

# @dataclass
# class Settings:
#     config_path: Path = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-strip-.yaml")
#     split: str = "train"  # train, val, test
#     max_events: int = -1  # -1 uses all available from config/directory
#     use_hit_eval: bool = True
#     output_dir: Path = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/analysis_plots_strip/")


# /share/rcif2/pduckett/hepattn-dq-pr/src/hepattn/experiments/trackml/configs/tracking-strip.yaml

def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _stats(x: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _plot_hist(values: np.ndarray, xlabel: str, title: str, out_path: Path) -> None:
    n_bins = int(np.clip(np.sqrt(values.size) * 2, 20, 80))
    if np.all(values == values.astype(np.int64)) and (values.max() - values.min()) <= 120:
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
    else:
        bins = np.linspace(values.min(), values.max(), n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=8, alpha=0.45)
    ax.set_xlabel("particles per event")
    ax.set_ylabel("hits per event")
    ax.set_title("Hits vs particles per event")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    split: str,
    n_events: int,
    hits_per_event: np.ndarray,
    particles_per_event: np.ndarray,
    event_max_num_particles: int,
) -> None:
    hit_stats = _stats(hits_per_event)
    part_stats = _stats(particles_per_event)
    capped = particles_per_event > event_max_num_particles

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["split", split])
        writer.writerow(["num_events", n_events])
        writer.writerow(["event_max_num_particles", event_max_num_particles])
        writer.writerow(["frac_events_over_particle_cap", float(np.mean(capped))])
        writer.writerow([])
        writer.writerow(["hits_per_event_min", hit_stats["min"]])
        writer.writerow(["hits_per_event_p05", hit_stats["p05"]])
        writer.writerow(["hits_per_event_median", hit_stats["median"]])
        writer.writerow(["hits_per_event_mean", hit_stats["mean"]])
        writer.writerow(["hits_per_event_p95", hit_stats["p95"]])
        writer.writerow(["hits_per_event_max", hit_stats["max"]])
        writer.writerow([])
        writer.writerow(["particles_per_event_min", part_stats["min"]])
        writer.writerow(["particles_per_event_p05", part_stats["p05"]])
        writer.writerow(["particles_per_event_median", part_stats["median"]])
        writer.writerow(["particles_per_event_mean", part_stats["mean"]])
        writer.writerow(["particles_per_event_p95", part_stats["p95"]])
        writer.writerow(["particles_per_event_max", part_stats["max"]])


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    settings = Settings()

    cfg = _read_yaml(settings.config_path)
    data_cfg = cfg["data"]

    split = settings.split
    split_dir = Path(data_cfg[f"{split}_dir"])
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    split_num = int(data_cfg.get(f"num_{split}", -1))
    num_events = split_num if split_num > 0 else -1
    hit_eval_path = data_cfg.get(f"hit_eval_{split}") if settings.use_hit_eval else None

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
        hit_eval_path=hit_eval_path,
        hit_filter_threshold=data_cfg.get("hit_filter_threshold", 0.1),
    )

    n_total = len(dataset)
    n_use = n_total if settings.max_events < 0 else min(settings.max_events, n_total)
    if n_use <= 0:
        raise RuntimeError("No events selected.")

    hits_per_event = np.zeros(n_use, dtype=np.int32)
    particles_per_event = np.zeros(n_use, dtype=np.int32)

    for i in range(n_use):
        hits, particles = dataset.load_event(i)
        hits_per_event[i] = len(hits)
        particles_per_event[i] = len(particles)

        if (i + 1) % 50 == 0 or (i + 1) == n_use:
            print(f"Processed {i + 1}/{n_use} events")

    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_hist(
        hits_per_event.astype(np.float64),
        xlabel="hits per event",
        title=f"{split}: distribution of hits per event",
        out_path=out_dir / f"{split}_hits_per_event_hist.png",
    )
    _plot_hist(
        particles_per_event.astype(np.float64),
        xlabel="particles per event",
        title=f"{split}: distribution of particles per event",
        out_path=out_dir / f"{split}_particles_per_event_hist.png",
    )
    _plot_scatter(
        particles_per_event.astype(np.float64),
        hits_per_event.astype(np.float64),
        out_path=out_dir / f"{split}_hits_vs_particles_scatter.png",
    )
    _write_summary(
        out_path=out_dir / f"{split}_summary.csv",
        split=split,
        n_events=n_use,
        hits_per_event=hits_per_event.astype(np.float64),
        particles_per_event=particles_per_event.astype(np.float64),
        event_max_num_particles=int(data_cfg["event_max_num_particles"]),
    )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
