"""Run the simplified old-format TrackML evaluator.

This script only supports paper-style `__test.h5` files and uses fixed
duplicate handling:

- identical-mask handling: `drop_from_metrics`
- same-majority handling: `postmatch_efficiency`
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

try:
    from eval_utils_simple import evaluate_file, format_summary, summarize_results
    from plot_utils import binned, profile_plot
except ImportError:  # pragma: no cover - supports `python -m ...`
    from hepattn.experiments.trackml.eval.eval_utils_simple import evaluate_file, format_summary, summarize_results
    from hepattn.experiments.trackml.eval.plot_utils import binned, profile_plot


# Edit these directly before running the script.
# `track_valid_threshold` and `iou_threshold` are optional per file.
EVAL_FILES = [
    {
        "name": "Paper",
        "fname": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5",
        "color": "tab:red",
        "prefilter_config": pathlib.Path("configs/tracking-eta4-pt900.yaml"),
        "prefilter_key_mode": "old",
    },
]
NUM_EVENTS = None
ETA_CUT = 4.0
PT_CUT = 1.0
TRUTH_REFERENCE_MODE = "post_filter"
DEFAULT_TRACK_VALID_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.0
OUT_DIR = pathlib.Path("./eff_plots_600_pre/")
PT_BINS = np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10], dtype=np.float32)
ETA_BINS = np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], dtype=np.float32)
QTY_BINS = {"pt": PT_BINS, "eta": ETA_BINS}
QTY_SYMBOLS = {"pt": "p_\\mathrm{T}", "eta": "\\eta"}
QTY_UNITS = {"pt": "[GeV]", "eta": ""}

plt.rcParams["figure.dpi"] = 400
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True


@lru_cache(maxsize=None)
def _load_data_config(config_path: pathlib.Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)["data"]


def _plot_efficiency_curves(results: list[dict]) -> None:
    if not results:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth_suffix = "" if TRUTH_REFERENCE_MODE == "post_filter" else f"_{TRUTH_REFERENCE_MODE}_truth"

    for qty in ("pt", "eta"):
        bins = QTY_BINS[qty]
        fig, ax = plt.subplots(figsize=(6, 4))
        names = []

        for result in results:
            names.append(result["name"])
            parts = result["parts"]
            reconstructable = parts["reconstructable"].to_numpy(dtype=bool)
            if not reconstructable.any():
                continue

            x = parts.loc[reconstructable, f"particle_{qty}"].to_numpy()
            y_dm = parts.loc[reconstructable, "metric_eff_dm"].to_numpy(dtype=bool)
            dm_eff, dm_err = binned(y_dm, x, bins, underflow=False, overflow=False, binomial=False)
            profile_plot(dm_eff, dm_err, bins, axes=ax, colour=result["color"], ls="solid")

            y_perfect = parts.loc[reconstructable, "metric_eff_perfect"].to_numpy(dtype=bool)
            perfect_eff, perfect_err = binned(y_perfect, x, bins, underflow=False, overflow=False, binomial=False)
            profile_plot(perfect_eff, perfect_err, bins, axes=ax, colour=result["color"], ls="dotted")

        ax.set_ylim(0.8, 1.1 )
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(rf"Particle ${QTY_SYMBOLS[qty]}^\mathrm{{True}}$ {QTY_UNITS[qty]}")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")

        if qty == "pt":
            ax.set_xlim([0, 10.5])
            ax.set_xticks(np.arange(start=2, stop=11, step=2))
        elif qty == "eta":
            ax.set_xlim([-4.5, 4.5])
            ax.set_xticks(np.arange(start=-4, stop=4.5, step=1))

        model_leg = ax.legend(
            handles=[Line2D([0], [0], color=result["color"], label=result["name"]) for result in results],
            frameon=False,
            loc="upper left",
        )
        ax.add_artist(model_leg)
        ax.legend(
            handles=[Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")],
            frameon=False,
            loc="upper right",
        )

        fig.savefig(OUT_DIR / f"{qty}_eff{truth_suffix}.pdf")
        plt.close(fig)

    # Eta efficiency plot restricted to 1 <= pT <= 10
    pt_lo, pt_hi = 2.0, 10.0
    bins = QTY_BINS["eta"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = []

    for result in results:
        names.append(result["name"])
        parts = result["parts"]
        reconstructable = parts["reconstructable"].to_numpy(dtype=bool)
        pt_mask = (parts["particle_pt"].to_numpy() >= pt_lo) & (parts["particle_pt"].to_numpy() <= pt_hi)
        mask = reconstructable & pt_mask
        if not mask.any():
            continue

        x = parts.loc[mask, "particle_eta"].to_numpy()
        y_dm = parts.loc[mask, "metric_eff_dm"].to_numpy(dtype=bool)
        dm_eff, dm_err = binned(y_dm, x, bins, underflow=False, overflow=False, binomial=False)
        profile_plot(dm_eff, dm_err, bins, axes=ax, colour=result["color"], ls="solid")

        y_perfect = parts.loc[mask, "metric_eff_perfect"].to_numpy(dtype=bool)
        perfect_eff, perfect_err = binned(y_perfect, x, bins, underflow=False, overflow=False, binomial=False)
        profile_plot(perfect_eff, perfect_err, bins, axes=ax, colour=result["color"], ls="dotted")

    ax.set_ylim(0.8, 1.1)
    ax.set_ylabel("Efficiency")
    ax.set_xlabel(rf"Particle $\eta^\mathrm{{True}}$")
    ax.set_title(rf"${pt_lo:.0f} \leq p_\mathrm{{T}} \leq {pt_hi:.0f}$ GeV")
    ax.set_xlim([-4.5, 4.5])
    ax.set_xticks(np.arange(start=-4, stop=4.5, step=1))
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    model_leg = ax.legend(
        handles=[Line2D([0], [0], color=result["color"], label=result["name"]) for result in results],
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(model_leg)
    ax.legend(
        handles=[Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")],
        frameon=False,
        loc="upper right",
    )

    fig.savefig(OUT_DIR / f"eta_eff_pt{pt_lo:.0f}_{pt_hi:.0f}{truth_suffix}.pdf")
    plt.close(fig)


def main() -> None:
    plot_results = []
    for eval_config in EVAL_FILES:
        name = eval_config["name"]
        fname = eval_config["fname"]
        color = eval_config["color"]
        track_valid_threshold = float(eval_config.get("track_valid_threshold", DEFAULT_TRACK_VALID_THRESHOLD))
        iou_threshold = float(eval_config.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
        prefilter_config = eval_config.get("prefilter_config")
        prefilter_data_config = _load_data_config(prefilter_config) if prefilter_config is not None else None
        tracks, parts = evaluate_file(
            fname=fname,
            num_events=NUM_EVENTS,
            eta_cut=ETA_CUT,
            pt_cut=PT_CUT,
            track_valid_threshold=track_valid_threshold,
            iou_threshold=iou_threshold,
            truth_reference_mode=TRUTH_REFERENCE_MODE,
            prefilter_data_config=prefilter_data_config,
            prefilter_key_mode=eval_config.get("prefilter_key_mode"),
        )
        summary = summarize_results(tracks, parts)
        print(
            format_summary(
                summary,
                fname=fname,
                eta_cut=ETA_CUT,
                pt_cut=PT_CUT,
                track_valid_threshold=track_valid_threshold,
                iou_threshold=iou_threshold,
            )
        )
        plot_results.append({"name": name, "color": color, "parts": parts})

    _plot_efficiency_curves(plot_results)


if __name__ == "__main__":
    main()
