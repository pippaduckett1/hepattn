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

# /share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-3l-900-epochs30-baseline-iou-hn-activations_20260531-T112327/ckpts/epoch=029-val_loss=0.44856_test_eval.h5

# Edit these directly before running the script.
# `track_valid_threshold` and `iou_threshold` are optional per file.
EVAL_FILES = [
    # {
    #     "name": "Baseline",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5",
    #     "color": "mediumseagreen",
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "Optimised",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-3l-900-epochs30-baseline-iou-hn-activations_20260610-T122030/ckpts/epoch=029-val_loss=0.47899__test.h5",
    #     "color": "cornflowerblue",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "MF-1.0",
    #     "pt_cut": 1.0,
    #     "eta_cut": 4,
    #     "fname": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5",
    #     "color": "tab:red",
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "MF-0.6",
    #     "pt_cut": 0.6,
    #     "eta_cut": 2.5,
    #     "fname": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-0.6GeV-2.5eta_20240926-T224200/ckpts/epoch=015-val_loss=2.44026__test.h5",
    #     "color": "tab:green",
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta2p5-pt600.yaml"
    #     ),
    # },
    # {
    #     "name": "Pix1.0 Fixed-Query Baseline",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5",
    #     "color": "tab:red",
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "Pix1.0 Fixed-Query Optimised",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-3l-900-epochs30-baseline-iou-hn-activations_20260610-T122030/ckpts/epoch=029-val_loss=0.47899__test.h5",
    #     "color": "tab:grey",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "Pix1.0 DQ+MA",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/TRK-CKPTS/MA-900/epoch=029-val_loss=0.16017__test.h5",
    #     "color": "tab:orange",
    #     "track_valid_threshold": 0.15,
    #     "iou_threshold": 0.4,
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt900-loss-weights.yaml"
    #     ),
    # },
    # {
    #     "name": "Pix1.0 DQ+LSCA",
    #     "pt_cut": 1.0,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-lca-eta4-900-th0p25-w64-epochs30_20260330-T174224/ckpts/epoch=029-val_loss=0.30343__test.h5",
    #     "color": "tab:blue",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path(
    #         "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-900-inference.yaml"
    #     ),
    # },
    # {
    #     "name": "Pix0.6 DQ+MA",
    #     "pt_cut": 0.6,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-pt600-epochs30_20260427-T113007/ckpts/epoch=028-val_loss=0.28455__test.h5",
    #     "color": "tab:green",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600.yaml"),
    # },
    # {
    #     "name": "Pix0.6 DQ+MA",
    #     "pt_cut": 0.6,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/TRK-CKPTS/LCA-600/epoch=029-val_loss=0.30247__test.h5",
    #     "color": "tab:green",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600.yaml"),
    # },
    # {
    #     "name": "Pix0.6 DQ+LSCA",
    #     "pt_cut": 0.6,
    #     "fname": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-lca-epochs30_20260419-T143422/ckpts/epoch=028-val_loss=0.32162__test.h5",
    #     "color": "tab:purple",
    #     "track_valid_threshold": 0.5,
    #     "iou_threshold": 0.5,
    #     "prefilter_config": pathlib.Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600.yaml"),
    # },
]
NUM_EVENTS = None
ETA_CUT = 4.0
DEFAULT_PT_CUT = 1.0
TRUTH_REFERENCE_MODE = "pre_filter"
PREFILTER_KEY_MODE = "old"
DEFAULT_TRACK_VALID_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.0
OUT_DIR = pathlib.Path("./eff_plots_dq_comparison_baseline/")
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


def _style_efficiency_axis(ax) -> None:
    ax.set_ylim(0.8, 1.1)
    ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_ylabel("Efficiency")


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

        _style_efficiency_axis(ax)
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

    _style_efficiency_axis(ax)
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
        pt_cut = float(eval_config.get("pt_cut", DEFAULT_PT_CUT))
        track_valid_threshold = float(eval_config.get("track_valid_threshold", DEFAULT_TRACK_VALID_THRESHOLD))
        iou_threshold = float(eval_config.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
        prefilter_config = eval_config.get("prefilter_config")
        prefilter_data_config = _load_data_config(prefilter_config) if prefilter_config is not None else None
        tracks, parts = evaluate_file(
            fname=fname,
            num_events=NUM_EVENTS,
            eta_cut=ETA_CUT,
            pt_cut=pt_cut,
            track_valid_threshold=track_valid_threshold,
            iou_threshold=iou_threshold,
            truth_reference_mode=TRUTH_REFERENCE_MODE,
            prefilter_data_config=prefilter_data_config,
            prefilter_key_mode=PREFILTER_KEY_MODE,
        )
        summary = summarize_results(tracks, parts)
        print(
            format_summary(
                summary,
                fname=fname,
                eta_cut=ETA_CUT,
                pt_cut=pt_cut,
                track_valid_threshold=track_valid_threshold,
                iou_threshold=iou_threshold,
            )
        )
        plot_results.append({"name": name, "color": color, "parts": parts})

    _plot_efficiency_curves(plot_results)


if __name__ == "__main__":
    main()
