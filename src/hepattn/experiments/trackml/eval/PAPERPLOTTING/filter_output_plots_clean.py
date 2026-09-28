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
import math
import pathlib

import numpy as np
import yaml
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 400
# plt.rcParams["text.usetex"] = True
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True


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
out_dir = "plots"


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
    "600 MeV eta 4": "/share/rcif2/pduckett/hepattn-dq/HF-pix-600MeV-eta4_20260214-T195401/epoch=070-val_loss=0.20153_val_eval.h5",
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
from hit_evaluate import load_events
from plot_utils import binned, profile_plot


# ------------------------------------------------------------------------------
# Cell 10 (code)
filtering_results = {}
num_events = None
threshold = 0.1
for name, fname in filtering_fnames.items():
    filter_threshold = filtering_configs[name]["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"][0]["init_args"]["threshold"]
    filtering_results[name] = load_events(fname=fname, randomize=num_events, write_inputs=None, write_parts=True, threshold=threshold)

    print("loaded")

print("loaded")




def threshold_for_efficiency(metrics: dict, target_eff: float = 0.99) -> float:
    # metrics["roc_eff"] is recall (efficiency); metrics["roc_eff_pur_thr"] are the thresholds for that curve
    eff = np.asarray(metrics["roc_eff"])
    thr = np.asarray(metrics["roc_eff_pur_thr"])

    # precision_recall_curve returns thr with length = len(eff) - 1
    # so align lengths by dropping the last eff entry
    eff = eff[:-1]

    i = int(np.argmin(np.abs(eff - target_eff)))
    return float(thr[i])

# usage:
for name, (_hits, _targets, _parts, metrics) in filtering_results.items():
    thr99 = threshold_for_efficiency(metrics, target_eff=0.99)
    print(f"{name}: threshold for 99% hit efficiency = {thr99:.4f}")



sys.exit()




# ------------------------------------------------------------------------------
# Cell 11 (markdown)
# ## Plotting metrics


# ------------------------------------------------------------------------------
# Cell 12 (markdown)
# ### Discriminant


# ------------------------------------------------------------------------------
# Cell 13 (code)
for name, (hits, targets, _parts, _metrics) in filtering_results.items():
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
    ax.plot(
        metrics["roc_eff"],
        metrics["roc_pur"],
        color=training_colours[name],
        label=f"{filtering_configs[name]['name']} {name}\nAUC: {metrics['roc_eff_pur_auc']:.4f}",
    )

    thid = np.argmin(np.abs(metrics["roc_eff_pur_thr"] - threshold))
    ax.scatter(metrics["roc_eff"][thid], metrics["roc_pur"][thid], color=training_colours[name], s=100)

ax.set_xlabel("Hit Efficiency")
ax.set_ylabel("Hit Purity")
ax.set_xlim(0.9, 1.01)
ax.set_ylim(0.3, 1.01)
ax.grid(which="both")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.grid(zorder=0, alpha=0.25, linestyle="--")
ax.legend()


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
    ax[0].plot(
        metrics["roc_eff"],
        metrics["roc_pur"],
        color=training_colours[name],
        label=f"{filtering_configs[name]['name']} {name}\nAUC: {metrics['roc_eff_pur_auc']:.4f}",
    )
    thid = np.argmin(np.abs(metrics["roc_eff_pur_thr"] - threshold))
    ax[0].scatter(metrics["roc_eff"][thid], metrics["roc_pur"][thid], color=training_colours[name], s=100)

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

fig.savefig(out_dir + "/filter_response.pdf")
plt.show()


# ------------------------------------------------------------------------------
# Cell 23 (code)
# calculate the threshold that gives 99% hit efficiency
for name, (_hits, _targets, _parts, metrics) in filtering_results.items():
    thid = np.argmin(np.abs(metrics["roc_eff"] - 0.99))
    print(f"{name} threshold for 99% hit efficiency: {metrics['roc_eff_pur_thr'][thid]:.4f}")


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
