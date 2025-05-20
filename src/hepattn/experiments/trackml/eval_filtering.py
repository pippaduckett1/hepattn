from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import h5py
from scipy.stats import binned_statistic
from tqdm import tqdm
import argparse

from hepattn.experiments.trackml.data import TrackMLDataset
from hepattn.utils.eval_plots import plot_hist_to_ax, bayesian_binomial_error
from hepattn.experiments.trackml.plot_event import plot_trackml_event_reconstruction

import torch
import sys

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1/(1 + np.exp(-np.clip(x, -10, 10)))


def main(config_path, hit_eval_path, particle_bins):

    # Now create the dataset
    config = yaml.safe_load(config_path.read_text())["data"]
    inputs = config["inputs"]

    # Add in extra target fields that will allow us to recompute reconstructability
    dirpath = config['test_dir']
    targets = config["targets"]
    targets["particle"] = ["pt", "eta", "phi"]
    num_events = 100
    hit_volume_ids = config['hit_volume_ids']
    particle_min_pt = config['particle_min_pt']
    particle_max_abs_eta = config['particle_max_abs_eta']
    particle_min_num_hits = config['particle_min_num_hits']
    event_max_num_particles = config['event_max_num_particles']

    dataset = TrackMLDataset(
        dirpath=dirpath,
        inputs=inputs,
        targets=targets,
        num_events=num_events,
        hit_volume_ids=hit_volume_ids,
        particle_min_pt=particle_min_pt,
        particle_max_abs_eta=particle_max_abs_eta,
        particle_min_num_hits=particle_min_num_hits,
        event_max_num_particles=event_max_num_particles,
        hit_eval_path=None)

    # Define bins for particle retention rate under the nominal working point
    particle_pre_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}
    particle_post_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}

    # Add arrays to track hit retention metrics
    total_hits_per_event = []
    valid_hits_per_event = []
    retained_valid_hits_per_event = []
    retained_noise_hits_per_event = []
    valid_hit_retention_rate = []
    noise_hit_retention_rate = []

    num_hits_pre = []
    num_recon_parts_pre = []
    num_hits_post = []
    num_recon_parts_post = []

    # Iterate over the events
    for idx in tqdm(range(num_events)):
    # for idx in range(len(dataset.event_names)):

        # Load the data from the event        
        # Note we are using load event, so evenerying is numpy arrays that are unbatched
        input, target = dataset[idx]
    
        with h5py.File(hit_eval_path, "r") as hit_eval_file:
            print(dataset.event_names[idx])
            hit_filter_pred = hit_eval_file[dataset.event_names[idx]]['preds']['final']['hit_filter']['hit_on_valid_particle'][0]

        # Record number of reconstructable particles and hits before filtering
        hit_valid_pre = target['hit_on_valid_particle']
        particle_valid_pre = target['particle_valid'][target['particle_valid']]
 
        num_hits_pre.append(sum(hit_valid_pre))
        num_recon_parts_pre.append(particle_valid_pre.sum())

        # Calculate hit retention metrics
        total_hits = hit_valid_pre.shape[1]
        valid_hits = hit_valid_pre.sum()
        noise_hits = total_hits - valid_hits
        
        retained_valid_hits = (hit_valid_pre & hit_filter_pred).sum()
        retained_noise_hits = (torch.tensor(hit_filter_pred) & ~hit_valid_pre).sum()
        
        total_hits_per_event.append(total_hits)
        valid_hits_per_event.append(valid_hits)
        retained_valid_hits_per_event.append(retained_valid_hits)
        retained_noise_hits_per_event.append(retained_noise_hits)
        
        valid_hit_retention_rate.append(retained_valid_hits / valid_hits if valid_hits > 0 else 0)
        noise_hit_retention_rate.append(retained_noise_hits / noise_hits if noise_hits > 0 else 0)

        filtered_hits_mask = hit_valid_pre & hit_filter_pred
        broadcast_mask = filtered_hits_mask.reshape(1, 1, -1)
        target["particle_hit_valid_masked"] = target["particle_hit_valid"] * broadcast_mask
        particles_hit_count = target["particle_hit_valid_masked"].sum(dim=-1)
        particle_recon_post = (particles_hit_count >= 3).bool()
        particle_recon_post = particle_recon_post[target["particle_valid"]]

        num_recon_parts_post.append(particle_recon_post.sum())
        num_hits_post.append(filtered_hits_mask.sum())

        # Fill the particle histograms
        for field, bins in particle_bins.items():
            particle_field = target[f"particle_{field}"][target["particle_valid"]]

            post_count, _, _ = binned_statistic(particle_field, particle_recon_post, statistic="sum", bins=bins)
            pre_count, _, _ = binned_statistic(particle_field, particle_valid_pre, statistic="sum", bins=bins)

            particle_pre_counts[field] += pre_count
            particle_post_counts[field] += post_count


    plot_save_dir = Path(__file__).resolve().parent / Path("evalplots")

    # Plot valid hit retention efficiency
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    # Plot 1: Valid hit retention rate per event
    events = np.arange(len(valid_hit_retention_rate))
    axs[0].scatter(events, valid_hit_retention_rate, label='Valid hits retained', alpha=0.7)
    axs[0].scatter(events, noise_hit_retention_rate, label='Noise hits retained', alpha=0.7)
    axs[0].set_xlabel('Event index')
    axs[0].set_ylabel('Retention rate')
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()
    axs[0].grid(alpha=0.25, linestyle='--')
    
    # Plot 2: Hit counts per event
    axs[1].bar(events, valid_hits_per_event, label='Valid hits', alpha=0.5)
    axs[1].bar(events, retained_valid_hits_per_event, label='Retained valid hits', alpha=0.7)
    axs[1].set_xlabel('Event index')
    axs[1].set_ylabel('Number of hits')
    axs[1].legend()
    axs[1].grid(alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    fig.savefig(plot_save_dir / Path("hit_retention_per_event.png"))
    
    # Plot hit retention summary statistics
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Histogram of valid hit retention rates
    axs[0].hist(valid_hit_retention_rate, bins=20, alpha=0.7, label='Valid hit retention')
    axs[0].hist(noise_hit_retention_rate, bins=20, alpha=0.7, label='Noise hit retention')
    axs[0].set_xlabel('Retention rate')
    axs[0].set_ylabel('Number of events')
    axs[0].legend()
    axs[0].grid(alpha=0.25, linestyle='--')
    
    # Plot 2: Scatter of retained valid hits vs. retained noise hits
    axs[1].scatter(retained_valid_hits_per_event, retained_noise_hits_per_event, alpha=0.7)
    axs[1].set_xlabel('Retained valid hits')
    axs[1].set_ylabel('Retained noise hits')
    axs[1].grid(alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    fig.savefig(plot_save_dir / Path("hit_retention_summary.png"))
    
    # Add summary statistics to console output
    avg_valid_retention = np.mean(valid_hit_retention_rate)
    avg_noise_retention = np.mean(noise_hit_retention_rate)
    print(f"Average valid hit retention rate: {avg_valid_retention:.4f}")
    print(f"Average noise hit retention rate: {avg_noise_retention:.4f}")
    print(f"Total valid hits: {sum(valid_hits_per_event)}")
    print(f"Total retained valid hits: {sum(retained_valid_hits_per_event)}")
    print(f"Total retained noise hits: {sum(retained_noise_hits_per_event)}")

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    frac_recon_parts_retained = np.array(num_recon_parts_post) / np.array(num_recon_parts_pre)
    ax.scatter(np.array(num_hits_post), frac_recon_parts_retained, alpha=0.5)

    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Number of Hits Retained")
    ax.set_ylabel("Fraction of Reconstructable Particles Retained")

    fig.savefig(plot_save_dir / Path("wp_scan.png"))

    pre_count = particle_pre_counts["pt"]
    post_count = particle_post_counts["pt"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["pt"], eff_errors)

    ax.set_xlabel("Truth Particle $p_T$ [GeV]")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_pt.png"))

    pre_count = particle_pre_counts["eta"]
    post_count = particle_post_counts["eta"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["eta"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\eta$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.96, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_eta.png"))

    pre_count = particle_pre_counts["phi"]
    post_count = particle_post_counts["phi"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["phi"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\phi$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_phi.png"))
    print("saving to ", plot_save_dir)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--config", type=Path, default=Path("/share/rcifdata/pduckett/hepattn/src/hepattn/experiments/trackml/configs/filtering.yaml"))
    args.add_argument("--hit_eval_path", type=Path, default=Path("/share/rcifdata/pduckett/hepattn/src/hepattn/experiments/trackml/logs/ec_eta4_20250415-T231339/ckpts/epoch=024-val_loss=0.21894_test_eval.h5"))
    args = args.parse_args()

    # Give the test eval file we are evaluating and setup the file
    particle_bins = {"pt": np.linspace(0.5, 10.0, 32), "eta": np.linspace(-4, 4, 32), "phi": np.linspace(-np.pi, np.pi, 32)}

    main(args.config, args.hit_eval_path, particle_bins)