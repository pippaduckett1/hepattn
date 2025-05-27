"""
Evaluation based on
https://arxiv.org/pdf/1904.06778

"""

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import stats


def load_event(fname, idx):
    """
    Load an event from an evaluation file and convert to DataFrame.
    """
    f = h5py.File(fname)
    g = f[f"event_{idx}"]

    # load unfiltered truth csv
    truth = pd.DataFrame({"pid": g["truth/particle_id"][:]})
    truth.index = truth.pid

    # load particles
    parts = pd.DataFrame({"pid": g["parts/pids"][:], "pt": g["parts/pts"][:], "eta": g["parts/etas"][:], "phi": g["parts/phis"][:], "vz": g["parts/vzs"][:]})
    parts.index = parts.pid

    # load hits
    hits = pd.DataFrame({"pid": g["hits/pids"][:]})
    hits.index = hits.pid

    # get masks
    masks = get_masks(g["preds/masks"][:])

    # load tracks (model outputs)
    tracks = pd.DataFrame({"class_pred": g["preds/class_preds"][:].argmax(-1)})
    tracks["n_assigned"] = masks.sum(-1)
    for k in g["preds/regression"]:
        tracks[k] = g[f"preds/regression/{k}"][:]
    tracks["pt"] = np.sqrt(tracks.px**2 + tracks.py**2)
    tracks["phi"] = np.arctan2(tracks.py, tracks.px)
    theta = np.arctan2(tracks.pt, tracks.pz)
    tracks["eta"] = -np.log(np.tan(0.5 * theta))

    # basic sanity checks
    assert len(np.unique(parts.pid) == len(parts)), "particle ids are not unique!"

    return hits, tracks, masks, parts, truth


def get_masks(masks):
    """Convert mask logits to binary masks."""
    masks = torch.from_numpy(masks).float().sigmoid().numpy()
    return masks > 0.5


def is_valid_track(assigned_hits, class_pred):
    """Cbeck whether an output track query is valid  - i.e. not a null output."""
    return assigned_hits >= 3 and class_pred == 0


def process_tracks(parts, hits, tracks, masks):
    """
    Loop over the output tracks from the model and compute metrics for each track.
    """

    # default - no match and not efficient
    tracks["eff_dm"] = False
    tracks["eff_lhc"] = False
    tracks["eff_perfect"] = False
    tracks["matched_pid"] = -1

    # loop over all output tracks
    matched_pis = []
    for reco_trk_idx in range(len(masks)):
        # check whether the model thinks this is a valid output, or just padding / invalid track
        is_valid = is_valid_track(tracks.n_assigned[reco_trk_idx], tracks.class_pred[reco_trk_idx])
        tracks.loc[reco_trk_idx, "valid"] = is_valid

        # if not valid, then don't try to match to a truth particle
        # have already set defaults so don't need to update here
        if not is_valid:
            continue

        # get the pids of hits assigned to this track
        mask = masks[reco_trk_idx]
        this_pids = hits.pid[mask]

        # find the most common pid, use this to define a truth particle
        majority_pid = int(stats.mode(this_pids, keepdims=False)[0])
        if majority_pid in matched_pis:
            continue
        tracks.loc[reco_trk_idx, "matched_pid"] = majority_pid
        matched_pis.append(majority_pid)

        # compute hit assignment metrics
        n_track_hits = len(this_pids)
        n_good_hits = len(this_pids[this_pids == majority_pid])
        n_truth_hits = len(hits.pid[hits.pid == majority_pid])
        recall = n_good_hits / n_truth_hits
        precision = n_good_hits / n_track_hits
        frac_out = (n_truth_hits - n_good_hits) / n_truth_hits

        # make a decision on efficiency
        eff_dm = recall > 0.5 and precision > 0.5
        eff_lhc = precision > 0.75
        eff_perfect = recall == 1 and precision == 1

        # append info
        tracks.loc[reco_trk_idx, "recall"] = recall
        tracks.loc[reco_trk_idx, "precision"] = precision
        tracks.loc[reco_trk_idx, "eff_dm"] = eff_dm
        tracks.loc[reco_trk_idx, "eff_lhc"] = eff_lhc
        tracks.loc[reco_trk_idx, "eff_perfect"] = eff_perfect
        if majority_pid == 0:
            tracks.loc[reco_trk_idx, "matched_pt"] = -1  # so we can see a spike on the histogram plot
            tracks.loc[reco_trk_idx, "matched_eta"] = np.nan
            tracks.loc[reco_trk_idx, "matched_phi"] = np.nan
            tracks.loc[reco_trk_idx, "matched_vz"] = np.nan
            tracks.loc[reco_trk_idx, "matched_reconstructable"] = False
        else:
            tracks.loc[reco_trk_idx, "matched_pt"] = parts.loc[majority_pid].pt
            tracks.loc[reco_trk_idx, "matched_eta"] = parts.loc[majority_pid].eta
            tracks.loc[reco_trk_idx, "matched_phi"] = parts.loc[majority_pid].phi
            tracks.loc[reco_trk_idx, "matched_vz"] = parts.loc[majority_pid].vz
            tracks.loc[reco_trk_idx, "matched_reconstructable"] = parts.loc[majority_pid].reconstructable

    # require that all pids and pts are unique - i.e. no multiple assignments of tracks to the same particle
    assert len(np.unique(tracks.matched_pid) == len(tracks.matched_pid))
    assert len(np.unique(tracks.matched_pt) == len(tracks.matched_pt))

    # second loop to identify duplicated tracks
    tracks["duplicate"] = False
    for i in range(len(masks)):
        if not tracks.valid[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not tracks.valid[j]:
                continue
            if (masks[i] == masks[j]).all():
                tracks.loc[i, "duplicate"] = True
                tracks.loc[j, "duplicate"] = True

            # mask_hash = hash_mask(masks[i])
            # tracks.loc[i, "mask_hash"] = mask_hash
            # tracks.loc[j, "mask_hash"] = mask_hash

    # might as well drop the null tracks here to simplify stuff downstream?
    tracks = tracks[tracks.valid]

    return tracks


def process_particles(parts, truth, eta_cut=2.5):
    """get reconstructable particles, using unfiltered hits"""

    # count number of hits left in selected detector volumns
    parts["n_hits"] = truth.pid.value_counts()

    # to be reconstructable we need at least 3 hits and eta < 2.5
    parts["reconstructable"] = (parts.n_hits >= 3) & (parts.eta.abs() < eta_cut)

    return parts


def eval_event(fname, idx, eta_cut=2.5):
    # load event
    hits, tracks, masks, parts, truth = load_event(fname, idx)

    # process particles
    parts = process_particles(parts, truth, eta_cut)

    # loop over model outputs
    tracks = process_tracks(parts, hits, tracks, masks)

    # add efficiency metric to particles
    dm_pid = tracks[tracks.eff_dm].matched_pid
    parts["eff_dm"] = parts.pid.isin(dm_pid)
    perfect_pid = tracks[tracks.eff_perfect].matched_pid
    parts["eff_perfect"] = parts.pid.isin(perfect_pid)
    lhc_pid = tracks[tracks.eff_lhc].matched_pid
    parts["eff_lhc"] = parts.pid.isin(lhc_pid)

    tracks["n_trk"] = tracks.valid.astype(int).sum()
    parts["n_vtx"] = parts.vz.nunique()
    tracks["event_idx"] = idx
    parts["event_idx"] = idx

    return tracks, parts


def eval_events(fname, num_events, eta_cut):
    # loop over several events
    for i in range(num_events):
        print(f"Processing event {i}", end="\r")
        if i == 0:
            tracks, parts = eval_event(fname, i, eta_cut)
        else:
            tracks_, parts_ = eval_event(fname, i, eta_cut)
            tracks = pd.concat([tracks, tracks_])
            parts = pd.concat([parts, parts_])

    # !! warning !! after this point, pid may not be unique
    tracks = tracks.drop(columns=["matched_pid"])
    parts = parts.drop(columns=["pid"])

    return parts, tracks