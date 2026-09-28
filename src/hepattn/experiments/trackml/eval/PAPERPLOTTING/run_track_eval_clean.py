import math
import pathlib
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, profile_plot
from track_eval_utils import load_events

# Disabled text.usetex: missing texlive font on Nikhef clusters.
plt.rcParams["figure.dpi"] = 400
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True


# ── Event / file utilities ────────────────────────────────────────────────────


def _event_id_to_event_index(event_id, idx_mode=None):
    """Convert an event identifier string/int to a numeric index."""
    event_str = str(event_id)
    if event_str.startswith("event_"):
        event_idx = int(event_str.removeprefix("event_"))
    else:
        event_idx = int(event_str)
    # Paper-format raw event parquet files are offset by 29800.
    if idx_mode == "paper":
        event_idx += 29800
    return event_idx


@lru_cache(maxsize=None)
def _event_file_map(test_dir, suffix):
    """Build a {event_index: path} map for parquet files matching event*-<suffix>.parquet."""
    event_map = {}
    for path in pathlib.Path(test_dir).glob(f"event*-{suffix}.parquet"):
        event_name = path.stem.removesuffix(f"-{suffix}")
        try:
            event_map[int(event_name.removeprefix("event"))] = path
        except ValueError:
            continue
    return event_map


def _postfilter_hits_per_event(fname, event_ids, eval_mode="new"):
    """Count hits per event after model-side hit filtering, read from the evaluation HDF5."""
    event_keys = [str(eid) for eid in event_ids]
    if not event_keys:
        return np.array([], dtype=np.int32)
    with h5py.File(fname, "r") as f:
        hit_counts = []
        for event_key in event_keys:
            if eval_mode == "old":
                hit_counts.append(np.array(f[event_key]["hits"]["pids"]).shape[0])
            else:
                hit_counts.append(np.array(f[event_key]["targets"]["particle_hit_valid"][:][0]).shape[-1])
    return np.asarray(hit_counts, dtype=np.int32)


def _prefilter_hits_per_event(data_config, event_ids, idx_mode="new"):
    """Count hits per event before hit filtering, read from raw parquet files."""
    if len(event_ids) == 0:
        return np.array([], dtype=np.int32)
    test_dir = pathlib.Path(data_config["test_dir"])
    hit_volume_ids = data_config.get("hit_volume_ids")
    event_file_map = _event_file_map(test_dir, "hits")
    hit_counts = []
    for event_id in event_ids:
        event_idx = _event_id_to_event_index(event_id, idx_mode=idx_mode)
        hits = pd.read_parquet(event_file_map[event_idx], columns=["volume_id"])
        if hit_volume_ids:
            hits = hits[hits["volume_id"].isin(hit_volume_ids)]
        hit_counts.append(len(hits))
    return np.asarray(hit_counts, dtype=np.int32)


def _load_prefilter_truth_particles_for_event(data_config, event_id, idx_mode="new"):
    """Load ground-truth particles for a single event, applying pre-filter acceptance cuts."""
    test_dir = pathlib.Path(data_config["test_dir"])
    event_idx = _event_id_to_event_index(event_id, idx_mode=idx_mode)
    particles = pd.read_parquet(_event_file_map(test_dir, "parts")[event_idx], columns=["particle_id", "px", "py", "pz"]).copy()
    hits = pd.read_parquet(_event_file_map(test_dir, "hits")[event_idx], columns=["particle_id", "volume_id"])

    hit_volume_ids = data_config.get("hit_volume_ids")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)]

    particles["particle_pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    particles["particle_phi"] = np.arctan2(particles["py"], particles["px"])
    particle_p = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["particle_eta"] = np.arctanh(particles["pz"] / particle_p)

    particles = particles[particles["particle_pt"] > data_config["particle_min_pt"]]
    particles = particles[particles["particle_eta"].abs() < data_config["particle_max_abs_eta"]]

    counts = hits["particle_id"].value_counts()
    keep_ids = counts[counts >= data_config["particle_min_num_hits"]].index.to_numpy()
    particles = particles[particles["particle_id"].isin(keep_ids)].copy()

    max_n = data_config["event_max_num_particles"]
    if len(particles) > max_n:
        if data_config.get("strict_max_objects", False):
            raise ValueError(f"Event {event_id} has {len(particles)} particles, but limit is {max_n}")
        particles = particles.iloc[:max_n].copy()

    particles["event_id"] = event_id
    particles["valid"] = True
    particles["reconstructable"] = True
    return particles[["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable"]]


def _attach_eval_particle_ids(fname, tracks, parts, eval_mode="new"):
    """Resolve eval-file integer slot indices to physical particle IDs for tracks and parts."""
    tracks = tracks.copy()
    parts = parts.copy()
    if len(parts) == 0:
        tracks["matched_particle_id"] = np.array([], dtype=np.int64)
        parts["particle_id"] = np.array([], dtype=np.int64)
        return tracks, parts

    parts_particle_ids = np.full(len(parts), -1, dtype=np.int64)
    tracks_particle_ids = np.full(len(tracks), -1, dtype=np.int64)
    particle_ids_by_event = {}

    with h5py.File(fname, "r") as f:
        for event_id, event_parts in parts.groupby("event_id", sort=False):
            event_key = str(event_id)
            if eval_mode == "old":
                particle_ids = np.array(f[event_key]["parts"]["pids"])
            else:
                particle_ids = np.array(f[event_key]["targets"]["particle_id"][:][0])
            if len(event_parts) > len(particle_ids):
                raise ValueError(f"Event {event_key} has more parts rows than particle IDs in the eval file.")
            parts_particle_ids[event_parts.index.to_numpy()] = particle_ids[: len(event_parts)]
            particle_ids_by_event[event_key] = parts_particle_ids[event_parts.index.to_numpy()]

    parts["particle_id"] = parts_particle_ids

    for event_id, event_tracks in tracks.groupby("event_id", sort=False):
        matched_pid = event_tracks["matched_pid"].to_numpy(dtype=np.int64)
        event_particle_ids = particle_ids_by_event[str(event_id)]
        valid_match = (matched_pid >= 0) & (matched_pid < len(event_particle_ids))
        matched_particle_ids = np.full(len(event_tracks), -1, dtype=np.int64)
        matched_particle_ids[valid_match] = event_particle_ids[matched_pid[valid_match]]
        tracks_particle_ids[event_tracks.index.to_numpy()] = matched_particle_ids

    tracks["matched_particle_id"] = tracks_particle_ids
    return tracks, parts


def _matched_truth_particle_ids(event_tracks, metric_column):
    """Return physical particle IDs credited by a metric, including recovered matches."""
    if event_tracks.empty:
        return set()

    metric_tracks = event_tracks.loc[event_tracks[metric_column]]
    matched_ids = set(metric_tracks["matched_particle_id"].astype(np.int64).tolist())

    if "out_of_acceptance_particle_id" in metric_tracks.columns:
        matched_ids.update(metric_tracks["out_of_acceptance_particle_id"].astype(np.int64).tolist())

    matched_ids.discard(-1)
    return matched_ids


def _build_prefilter_truth_reference_parts(
    data_config,
    fname,
    tracks,
    parts,
    eval_mode="new",
    idx_mode="new",
    key_mode=None,
):
    """Build a truth-particle reference DataFrame using pre-filter acceptance as the denominator.

    Pre-filter truth gives a fairer efficiency denominator: it includes all reconstructable
    particles that the model *could* have seen, not just those that survived hit filtering.
    """
    if key_mode is not None:
        eval_mode = "old" if key_mode == "old" else "new"
        idx_mode = "paper" if key_mode == "old" else "new"

    tracks_with_ids, _ = _attach_eval_particle_ids(fname, tracks, parts, eval_mode=eval_mode)
    truth_parts = []

    for event_id in np.sort(parts["event_id"].unique()):
        event_truth = _load_prefilter_truth_particles_for_event(data_config, event_id, idx_mode=idx_mode)
        event_tracks = tracks_with_ids[tracks_with_ids["event_id"] == event_id]

        dm_ids = _matched_truth_particle_ids(event_tracks, "eff_dm")
        perfect_ids = _matched_truth_particle_ids(event_tracks, "eff_perfect")

        event_truth["eff_dm"] = event_truth["particle_id"].isin(dm_ids)
        event_truth["eff_perfect"] = event_truth["particle_id"].isin(perfect_ids)
        truth_parts.append(event_truth)

    if not truth_parts:
        return pd.DataFrame(columns=["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable", "eff_dm", "eff_perfect"])
    return pd.concat(truth_parts, ignore_index=True)


def _fake_rate_excluding_duplicates(tracks, use_reconstructable_match):
    """Compute fake rate over non-duplicate tracks only."""
    unique_tracks = tracks.loc[~tracks["duplicate"]]
    if use_reconstructable_match:
        return (~(unique_tracks["eff_dm"] & unique_tracks["reconstructable_parts"])).mean()
    return (~unique_tracks["eff_dm"]).mean()


def _fake_rate_old_run_style(tracks):
    """Compute fake rate with the old evaluator convention.

    Duplicates remain in the valid-track denominator and are counted as fake
    because their `eff_dm` flag has already been cleared by the matcher.
    """
    return (~tracks["eff_dm"]).mean()


def _fake_rate_for_view(tracks_view, truth_match_column=None):
    """Compute fake rate for a duplicate-mode view.

    `metric_included` controls the denominator. `metric_fake_numerator_dm`
    controls which tracks are allowed to contribute to the fake numerator.
    This lets us reproduce the old notebook convention where identical-mask
    duplicates stay in the denominator but are excluded from the numerator.
    """
    included = tracks_view["metric_included"].to_numpy(dtype=bool)
    fake_numerator = tracks_view["metric_fake_numerator_dm"].to_numpy(dtype=bool)
    eff_dm = tracks_view["metric_eff_dm"].to_numpy(dtype=bool)

    if truth_match_column is None:
        fake_mask = (~eff_dm) & fake_numerator
    else:
        truth_match = tracks_view[truth_match_column].to_numpy(dtype=bool)
        fake_mask = (~(eff_dm & truth_match)) & fake_numerator

    return fake_mask[included].mean()


def _truth_acceptance_masks(truth_parts, eta_cut, pt_cut):
    """Return reconstructable masks with and without the truth pT cut."""
    no_pt = truth_parts["valid"].to_numpy(dtype=bool).copy()
    if "particle_eta" in truth_parts.columns:
        no_pt &= truth_parts["particle_eta"].abs().to_numpy() < eta_cut

    with_pt = no_pt.copy()
    if "particle_pt" in truth_parts.columns:
        with_pt &= truth_parts["particle_pt"].to_numpy() > pt_cut
    return with_pt, no_pt


def _build_duplicate_mode_views(tracks, truth_parts, mask_duplicate_handling, majority_particle_handling):
    """Derive per-track and per-particle metrics for a duplicate-handling convention."""
    tracks_view = tracks.reset_index(drop=True).copy()
    truth_view = truth_parts.reset_index(drop=True).copy()

    duplicate_base = tracks_view["duplicate"].to_numpy(dtype=bool)
    eff_dm_raw = tracks_view["eff_dm_raw"].to_numpy(dtype=bool)
    eff_perfect_raw = tracks_view["eff_perfect_raw"].to_numpy(dtype=bool)
    eff_lhc_raw = tracks_view["eff_lhc_raw"].to_numpy(dtype=bool)
    first_claim_match = tracks_view["first_claim_match"].to_numpy(dtype=bool)
    included_in_metrics = np.ones(len(tracks_view), dtype=bool)
    fake_numerator_dm = np.ones(len(tracks_view), dtype=bool)
    fake_numerator_perfect = np.ones(len(tracks_view), dtype=bool)
    fake_numerator_lhc = np.ones(len(tracks_view), dtype=bool)

    if mask_duplicate_handling == "count_as_fake":
        duplicate_dm = duplicate_base.copy()
        duplicate_perfect = duplicate_base.copy()
        duplicate_lhc = duplicate_base.copy()
    elif mask_duplicate_handling == "old_notebook_style":
        duplicate_dm = duplicate_base.copy()
        duplicate_perfect = duplicate_base.copy()
        duplicate_lhc = duplicate_base.copy()
        fake_numerator_dm &= ~duplicate_base
        fake_numerator_perfect &= ~duplicate_base
        fake_numerator_lhc &= ~duplicate_base
    elif mask_duplicate_handling == "drop_from_metrics":
        duplicate_dm = np.zeros(len(tracks_view), dtype=bool)
        duplicate_perfect = np.zeros(len(tracks_view), dtype=bool)
        duplicate_lhc = np.zeros(len(tracks_view), dtype=bool)
        included_in_metrics &= ~duplicate_base
    else:
        raise ValueError(f"Unknown mask_duplicate_handling={mask_duplicate_handling!r}")

    if majority_particle_handling == "first_claim":
        eff_dm = eff_dm_raw & first_claim_match
        eff_perfect = eff_perfect_raw & first_claim_match
        eff_lhc = eff_lhc_raw & first_claim_match
    elif majority_particle_handling == "postmatch_efficiency":
        eff_dm = eff_dm_raw.copy()
        eff_perfect = eff_perfect_raw.copy()
        eff_lhc = eff_lhc_raw.copy()

        for (_, majority_pid), group in tracks_view.groupby(["event_id", "majority_particle_id"], sort=False):
            if int(majority_pid) <= 0:
                continue
            group_idx = group.index.to_numpy(dtype=np.int64)[included_in_metrics[group.index.to_numpy(dtype=np.int64)]]
            if group_idx.size == 0:
                continue

            dm_candidates = group_idx[eff_dm[group_idx]]
            if dm_candidates.size > 0:
                keep = dm_candidates[0]
                duplicate_dm[group_idx[group_idx != keep]] = True
                eff_dm[group_idx] = False
                eff_dm[keep] = True

            perfect_candidates = group_idx[eff_perfect[group_idx]]
            if perfect_candidates.size > 0:
                keep = perfect_candidates[0]
                duplicate_perfect[group_idx[group_idx != keep]] = True
                eff_perfect[group_idx] = False
                eff_perfect[keep] = True

            lhc_candidates = group_idx[eff_lhc[group_idx]]
            if lhc_candidates.size > 0:
                keep = lhc_candidates[0]
                duplicate_lhc[group_idx[group_idx != keep]] = True
                eff_lhc[group_idx] = False
                eff_lhc[keep] = True
    else:
        raise ValueError(f"Unknown majority_particle_handling={majority_particle_handling!r}")

    if mask_duplicate_handling == "count_as_fake":
        eff_dm &= ~duplicate_base
        eff_perfect &= ~duplicate_base
        eff_lhc &= ~duplicate_base

    tracks_view["metric_eff_dm"] = eff_dm
    tracks_view["metric_eff_perfect"] = eff_perfect
    tracks_view["metric_eff_lhc"] = eff_lhc
    tracks_view["metric_duplicate_dm"] = duplicate_dm
    tracks_view["metric_duplicate_perfect"] = duplicate_perfect
    tracks_view["metric_duplicate_lhc"] = duplicate_lhc
    tracks_view["metric_included"] = included_in_metrics
    tracks_view["metric_fake_numerator_dm"] = fake_numerator_dm
    tracks_view["metric_fake_numerator_perfect"] = fake_numerator_perfect
    tracks_view["metric_fake_numerator_lhc"] = fake_numerator_lhc

    dm_keys = set(
        zip(
            tracks_view.loc[tracks_view["metric_eff_dm"] & tracks_view["metric_included"], "event_id"],
            tracks_view.loc[tracks_view["metric_eff_dm"] & tracks_view["metric_included"], "majority_particle_id"].astype(np.int64),
        )
    )
    perfect_keys = set(
        zip(
            tracks_view.loc[tracks_view["metric_eff_perfect"] & tracks_view["metric_included"], "event_id"],
            tracks_view.loc[tracks_view["metric_eff_perfect"] & tracks_view["metric_included"], "majority_particle_id"].astype(np.int64),
        )
    )
    lhc_keys = set(
        zip(
            tracks_view.loc[tracks_view["metric_eff_lhc"] & tracks_view["metric_included"], "event_id"],
            tracks_view.loc[tracks_view["metric_eff_lhc"] & tracks_view["metric_included"], "majority_particle_id"].astype(np.int64),
        )
    )
    truth_keys = list(zip(truth_view["event_id"], truth_view["particle_id"].astype(np.int64)))
    truth_view["metric_eff_dm"] = [key in dm_keys for key in truth_keys]
    truth_view["metric_eff_perfect"] = [key in perfect_keys for key in truth_keys]
    truth_view["metric_eff_lhc"] = [key in lhc_keys for key in truth_keys]

    return tracks_view, truth_view


def _track_truth_acceptance_flags(tracks_view, truth_view, eta_cut, pt_cut):
    """Attach truth-acceptance flags for a track's majority particle, with and without a pT cut."""
    with_pt_mask, no_pt_mask = _truth_acceptance_masks(truth_view, eta_cut=eta_cut, pt_cut=pt_cut)
    lookup = {
        (event_id, int(particle_id)): (bool(with_pt), bool(no_pt))
        for event_id, particle_id, with_pt, no_pt in zip(
            truth_view["event_id"],
            truth_view["particle_id"],
            with_pt_mask,
            no_pt_mask,
        )
    }

    matched_with_pt = np.zeros(len(tracks_view), dtype=bool)
    matched_no_pt = np.zeros(len(tracks_view), dtype=bool)
    for row_idx, (event_id, majority_pid) in enumerate(
        zip(tracks_view["event_id"], tracks_view["majority_particle_id"].astype(np.int64), strict=False)
    ):
        matched_with_pt[row_idx], matched_no_pt[row_idx] = lookup.get((event_id, int(majority_pid)), (False, False))
    return matched_with_pt, matched_no_pt


# ── Threshold utilities ───────────────────────────────────────────────────────


def _threshold_tag(track_valid_threshold, iou_threshold):
    """Short string identifier for a threshold pair, used in output filenames."""
    tv = f"{track_valid_threshold:.2f}".replace(".", "p")
    iou = f"{iou_threshold:.2f}".replace(".", "p")
    return f"tv{tv}_iou{iou}"


def _as_threshold_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    return [float(value)]


def _build_threshold_configs(track_valid_thresholds, iou_thresholds):
    """Return all (track_valid_threshold, iou_threshold) combinations as a list of dicts."""
    return [
        {"track_valid_threshold": tv, "iou_threshold": iou}
        for tv in _as_threshold_list(track_valid_thresholds)
        for iou in _as_threshold_list(iou_thresholds)
    ]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_all_results(
    tracking_fnames,
    tracking_configs,
    default_track_valid_thresholds,
    default_iou_thresholds,
    tracking_threshold_overrides,
    hit_order_by_name,
    num_events,
    random_seed,
    particle_targets,
    truth_reference_mode,
    match_min_hits,
    match_min_pt,
    event_cache_dir,
    reuse_saved_event_cache,
    write_event_cache,
):
    """Load tracking results for every model × threshold combination.

    Each model uses its own threshold sweep. Per-model overrides for
    track_valid_thresholds and iou_thresholds can be provided via
    tracking_threshold_overrides; models not listed fall back to the
    default_track_valid_thresholds / default_iou_thresholds.

    Returns two dicts keyed by (track_valid_threshold, iou_threshold):
      tracking_results[key][name]       -> (tracks_df, parts_df)
      truth_reference_results[key][name] -> truth_parts_df
    """
    # Build per-model threshold configs, falling back to global defaults.
    model_threshold_configs = {}
    for name in tracking_fnames:
        overrides = tracking_threshold_overrides.get(name, {})
        tvs = overrides.get("track_valid_thresholds", default_track_valid_thresholds)
        ious = overrides.get("iou_thresholds", default_iou_thresholds)
        model_threshold_configs[name] = _build_threshold_configs(tvs, ious)

    # Initialise result dicts for every unique threshold combination across all models.
    all_threshold_keys = {
        (cfg["track_valid_threshold"], cfg["iou_threshold"])
        for cfgs in model_threshold_configs.values()
        for cfg in cfgs
    }
    tracking_results = {key: {} for key in all_threshold_keys}
    truth_reference_results = {key: {} for key in all_threshold_keys}

    for name, fname in tracking_fnames.items():
        data_cfg = tracking_configs[name]["data"]
        eta_cut = data_cfg["particle_max_abs_eta"]
        pt_cut = data_cfg["particle_min_pt"]
        # Distinguish eval-file structure from raw-event parquet indexing.
        overrides = tracking_threshold_overrides.get(name, {})
        eval_mode = overrides.get("eval_mode", "old" if name == "Paper" else "new")
        idx_mode = overrides.get("idx_mode", "paper" if eval_mode == "old" else "new")

        for cfg in model_threshold_configs[name]:
            tv, iou = cfg["track_valid_threshold"], cfg["iou_threshold"]
            threshold_key = (tv, iou)
            print(f"  {name}: tv={tv:.2f}, iou={iou:.2f}, pT > {pt_cut}, |η| < {eta_cut}")

            cache_key = f"{name.replace(' ', '_').lower()}_{_threshold_tag(tv, iou)}"
            tracking_results[threshold_key][name] = load_events(
                fname=fname,
                pt_cut=pt_cut,
                eta_cut=eta_cut,
                index_list=None,
                randomize=None,
                random_seed=None,
                particle_targets=particle_targets,
                regression=False,
                eval_mode=eval_mode,
                track_valid_threshold=tv,
                iou_threshold=iou,
                match_min_hits=match_min_hits,
                match_min_pt=match_min_pt,
                hit_order_mode=hit_order_by_name.get(name, "auto"),
                cache_dir=event_cache_dir,
                cache_key=cache_key,
                reuse_saved=reuse_saved_event_cache,
                write_cache=write_event_cache,
            )
            tracks, parts = tracking_results[threshold_key][name]

            if truth_reference_mode == "pre_filter":
                # Denominator uses truth particles from raw parquet, before hit filtering.
                truth_reference_results[threshold_key][name] = _build_prefilter_truth_reference_parts(
                    data_config=data_cfg,
                    fname=fname,
                    tracks=tracks,
                    parts=parts,
                    eval_mode=eval_mode,
                    idx_mode=idx_mode,
                )
            else:
                # Denominator uses truth particles after hit filtering (post_filter).
                truth_reference_results[threshold_key][name] = parts.copy()

    print("Loaded all events.")
    return tracking_results, truth_reference_results


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_efficiency(
    tracking_results,
    truth_reference_results,
    tracking_fnames,
    particle_targets,
    qty_bins,
    qty_symbols,
    qty_units,
    training_colours,
    out_dir,
    truth_reference_mode="post_filter",
):
    """Plot DM and perfect-match efficiency vs kinematic quantities for all threshold combinations."""
    single_combo = len(tracking_results) == 1
    truth_suffix = "" if truth_reference_mode == "post_filter" else f"_{truth_reference_mode}_truth"

    for (tv, iou), threshold_results in tracking_results.items():
        threshold_suffix = "" if single_combo else "_" + _threshold_tag(tv, iou)

        for qty in [q for q in particle_targets if q in {"pt", "eta", "phi"}]:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = []

            for name, (_tracks, _parts) in threshold_results.items():
                if name not in tracking_fnames:
                    continue
                names.append(name)

                truth_parts = truth_reference_results[(tv, iou)][name]
                reconstructable = truth_parts["reconstructable"]
                x = truth_parts["particle_" + qty][reconstructable]

                # Double majority efficiency (solid)
                bc, be = binned(truth_parts["eff_dm"][reconstructable], x, qty_bins[qty], underflow=False, overflow=False, binomial=False)
                profile_plot(bc, be, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

                # Perfect-match efficiency (dotted)
                bc, be = binned(truth_parts["eff_perfect"][reconstructable], x, qty_bins[qty], underflow=False, overflow=False, binomial=False)
                profile_plot(bc, be, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

            ax.set_ylim(0.8, 1.04)
            ax.set_ylabel("Efficiency")
            ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            if qty == "pt":
                ax.set_xlim([0, 10.5])
                ax.set_xticks(np.arange(start=2, stop=11, step=2))
            elif qty == "eta":
                ax.set_xlim([-4.5, 4.5])
                ax.set_xticks(np.arange(start=-4, stop=4.5, step=1))
            elif qty == "phi":
                ax.set_xlim([-3.5, 3.5])
                ax.set_xticks(np.arange(start=-3, stop=3.5, step=1))

            # Two independent legends: one for models (colour), one for efficiency definition (line style)
            model_leg = ax.legend(
                handles=[Line2D([0], [0], color=training_colours[n], label=n) for n in names],
                frameon=False, loc="upper left",
            )
            ax.add_artist(model_leg)
            ax.legend(
                handles=[Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")],
                frameon=False, loc="upper right",
            )

            fig.savefig(out_dir + f"{qty}_eff{threshold_suffix}{truth_suffix}.png")
            plt.close(fig)


# ── Summary metrics ───────────────────────────────────────────────────────────


def print_and_save_summary(
    tracking_results,
    truth_reference_results,
    tracking_configs,
    tracking_fnames,
    tracking_threshold_overrides,
    truth_reference_mode,
    out_dir,
):
    """Print per-model efficiency / fake-rate metrics and save a summary CSV."""
    summary_rows = []
    mask_duplicate_modes = {
        "count_as_fake": "identical-mask duplicates stay in denominator and count as fake",
        "old_notebook_style": "identical-mask duplicates stay in denominator but are excluded from the fake-rate numerator",
        "drop_from_metrics": "identical-mask duplicates are removed from metrics entirely",
    }
    majority_particle_modes = {
        "first_claim": "first majority claimant wins before efficiency matching",
        "postmatch_efficiency": "choose first efficient track after efficiency matching",
    }

    for (tv, iou), threshold_results in tracking_results.items():
        print(f"\nThreshold: track_valid={tv:.2f}, iou={iou:.2f}")

        for name, (tracks, parts) in threshold_results.items():
            overrides = tracking_threshold_overrides.get(name, {})
            eval_mode = overrides.get("eval_mode", "old" if name == "Paper" else "new")
            idx_mode = overrides.get("idx_mode", "paper" if eval_mode == "old" else "new")
            eta_cut = tracking_configs[name]["data"]["particle_max_abs_eta"]
            pt_cut = tracking_configs[name]["data"]["particle_min_pt"]
            event_ids = np.sort(parts["event_id"].unique())
            n_events = len(event_ids)

            truth_parts = truth_reference_results[(tv, iou)][name]
            valid_parts_per_event = parts[parts.valid].groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
            tracks_per_event = tracks.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
            pre_hits = _prefilter_hits_per_event(tracking_configs[name]["data"], event_ids, idx_mode=idx_mode)
            post_hits = _postfilter_hits_per_event(tracking_fnames[name], event_ids, eval_mode=eval_mode)

            def _pct(x):
                return f"{x * 100:.3g}%"

            print(f"\n  {name}")
            print(f"  N events: {n_events}")
            print(f"  Post-filter valid truth particles:  total={len(parts[parts.valid])}, mean/event={valid_parts_per_event.mean():.3f} ± {valid_parts_per_event.std():.3f}")
            print(f"  Valid predicted tracks:             total={len(tracks)}, mean/event={tracks_per_event.mean():.3f} ± {tracks_per_event.std():.3f}")
            print(f"  Hits (pre-filter):  total={pre_hits.sum()}, mean/event={pre_hits.mean():.3f} ± {pre_hits.std():.3f}")
            print(f"  Hits (post-filter): total={post_hits.sum()}, mean/event={post_hits.mean():.3f} ± {post_hits.std():.3f}")

            for mask_duplicate_mode, mask_desc in mask_duplicate_modes.items():
                for majority_particle_mode, majority_desc in majority_particle_modes.items():
                    tracks_view, truth_view = _build_duplicate_mode_views(
                        tracks,
                        truth_parts,
                        mask_duplicate_handling=mask_duplicate_mode,
                        majority_particle_handling=majority_particle_mode,
                    )
                    ref_with_pt_mask, ref_no_pt_mask = _truth_acceptance_masks(truth_view, eta_cut=eta_cut, pt_cut=pt_cut)
                    ref_with_pt = truth_view[ref_with_pt_mask]
                    ref_no_pt = truth_view[ref_no_pt_mask]
                    ref_with_pt_per_event = ref_with_pt.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
                    ref_no_pt_per_event = ref_no_pt.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()

                    matched_truth_with_pt, matched_truth_no_pt = _track_truth_acceptance_flags(
                        tracks_view,
                        truth_view,
                        eta_cut=eta_cut,
                        pt_cut=pt_cut,
                    )
                    tracks_view["metric_truth_with_pt_cut"] = matched_truth_with_pt
                    tracks_view["metric_truth_without_pt_cut"] = matched_truth_no_pt

                    included = tracks_view["metric_included"].to_numpy(dtype=bool)
                    fake_rate_with_pt = _fake_rate_for_view(tracks_view, truth_match_column="metric_truth_with_pt_cut")
                    fake_rate_without_pt = _fake_rate_for_view(tracks_view, truth_match_column="metric_truth_without_pt_cut")
                    fake_rate_all_included_tracks = _fake_rate_for_view(tracks_view)

                    eff_with_pt = ref_with_pt["metric_eff_dm"].mean()
                    eff_without_pt = ref_no_pt["metric_eff_dm"].mean()
                    perf_with_pt = ref_with_pt["metric_eff_perfect"].mean()
                    perf_without_pt = ref_no_pt["metric_eff_perfect"].mean()

                    print(f"")
                    print(f"    Identical-mask handling: {mask_duplicate_mode} ({mask_desc})")
                    print(f"    Same-majority handling:  {majority_particle_mode} ({majority_desc})")
                    print(
                        f"    Reference truth particles ({truth_reference_mode}, with truth pT cut): "
                        f"total={len(ref_with_pt)}, mean/event={ref_with_pt_per_event.mean():.3f} ± {ref_with_pt_per_event.std():.3f}"
                    )
                    print(
                        f"    Reference truth particles ({truth_reference_mode}, no truth pT cut):   "
                        f"total={len(ref_no_pt)}, mean/event={ref_no_pt_per_event.mean():.3f} ± {ref_no_pt_per_event.std():.3f}"
                    )
                    print(f"    DM eff (with truth pT cut):          {_pct(eff_with_pt)}")
                    print(f"    DM eff (without truth pT cut):       {_pct(eff_without_pt)}")
                    print(f"    Perfect eff (with truth pT cut):     {_pct(perf_with_pt)}")
                    print(f"    Perfect eff (without truth pT cut):  {_pct(perf_without_pt)}")
                    print(f"    Fake rate (with truth pT cut):       {_pct(fake_rate_with_pt)}")
                    print(f"    Fake rate (without truth pT cut):    {_pct(fake_rate_without_pt)}")
                    print(f"    Fake rate (all included tracks):     {_pct(fake_rate_all_included_tracks)}")
                    print(f"    Excluded identical-mask duplicate rate: {_pct((~included).mean())}")
                    print(f"    Duplicate rate (DM):                 {_pct(tracks_view['metric_duplicate_dm'].mean())}")
                    print(f"    Duplicate rate (Perfect):            {_pct(tracks_view['metric_duplicate_perfect'].mean())}")

                    summary_rows.append({
                        "model": name,
                        "truth_reference_mode": truth_reference_mode,
                        "mask_duplicate_handling": mask_duplicate_mode,
                        "majority_particle_handling": majority_particle_mode,
                        "track_valid_threshold": tv,
                        "iou_threshold": iou,
                        "n_events": n_events,
                        "reference_truth_particles_with_truth_pt_cut": len(ref_with_pt),
                        "reference_truth_particles_without_truth_pt_cut": len(ref_no_pt),
                        "integrated_efficiency_with_truth_pt_cut": float(eff_with_pt),
                        "integrated_efficiency_without_truth_pt_cut": float(eff_without_pt),
                        "perfect_efficiency_with_truth_pt_cut": float(perf_with_pt),
                        "perfect_efficiency_without_truth_pt_cut": float(perf_without_pt),
                        "fake_rate_with_truth_pt_cut": float(fake_rate_with_pt),
                        "fake_rate_without_truth_pt_cut": float(fake_rate_without_pt),
                        "fake_rate_all_included_tracks": float(fake_rate_all_included_tracks),
                        "excluded_identical_mask_duplicate_rate": float((~included).mean()),
                        "duplicate_rate_dm": float(tracks_view["metric_duplicate_dm"].mean()),
                        "duplicate_rate_perfect": float(tracks_view["metric_duplicate_perfect"].mean()),
                    })

    summary_name = "track_eval_threshold_summary.csv" if truth_reference_mode == "post_filter" else f"track_eval_threshold_summary_{truth_reference_mode}_truth.csv"
    summary_path = pathlib.Path(out_dir) / summary_name
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote threshold summary to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Model checkpoints ────────────────────────────────────────────────────

    tracking_fnames = {
        "Paper": "/share/rcifdata/svanstroud/hepformer/hepformer/tracking/logs/HF-final-1GeV-hc0.1-eta4_20250307-T174811/ckpts/epoch=028-val_loss=1.53465__test.h5",

        "LCA 900": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-lca-eta4-900-th0p25-w64-epochs30_20260330-T174224/ckpts/epoch=029-val_loss=0.30343__test.h5",

        # "MA 900": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-3l-900-loss-weights-25-epochs30_20260329-T095012/ckpts/epoch=029-val_loss=0.16017_test_eval_.h5",
        # "LCA 900": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-lca-eta4-900-th0p25-w64-qual-epochs_20260408-T153434/ckpts/epoch=029-val_loss=0.28273_test_eval-w32.h5",

        # "LSCA 600": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-lca-3700-0p3-lw100-epochs30_20260504-T042408/ckpts/epoch=029-val_loss=0.29219_test_eval.h5",
        # "MA 600": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/TRK-v8-eta4-pt600-epochs30_20260427-T113007/ckpts/epoch=028-val_loss=0.28455_test_eval.h5",

    }


    tracking_config_fnames = {
        "Paper": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml",
        "MA 600": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt600.yaml",
        "LSCA 600": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600-inference.yaml",
        "LSCA 600 10epochs": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600-inference.yaml",
        "LSCA 600 3700": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-lca-eta4-600-inference.yaml",
        "MA 900": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml",
        "LCA 900": "/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/configs/tracking-eta4-pt1.yaml",
    }

    tracking_configs = {}
    for name, cfg_path in tracking_config_fnames.items():
        with pathlib.Path(cfg_path).open() as f:
            cfg = yaml.safe_load(f)
        print(f"  {name}: pT > {cfg['data']['particle_min_pt']}, |η| < {cfg['data']['particle_max_abs_eta']}")
        tracking_configs[name] = cfg

    # ── Visual style ─────────────────────────────────────────────────────────
    training_colours = {
        "Paper": "tab:green",
        "MA 600": "tab:orange",
        "LSCA 600": "tab:blue",
        "MA 900": "tab:red",
        "LCA 900": "tab:orange",
    }

    qty_bins = {
        "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
        "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
        "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
    }
    qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi"}
    qty_units = {"pt": "[GeV]", "eta": "", "phi": ""}

    # ── Evaluation parameters ─────────────────────────────────────────────────
    out_dir = ""
    event_cache_dir = pathlib.Path(out_dir) / "event_cache"
    num_events = None
    random_seed = None
    particle_targets = ["pt", "eta", "phi"]
    # "post_filter": efficiency denominator uses truth particles after hit filtering
    # "pre_filter":  efficiency denominator uses all reconstructable truth particles (fairer)
    truth_reference_mode = "post_filter"

    # Pre-matching cuts applied to truth particles before track-to-particle assignment.
    # Predicted tracks whose best-matching truth particle fails either cut are treated
    # as unmatched (and counted as fakes if they are otherwise valid).
    # These cuts are intentionally looser than the reconstructability cuts (pt_cut, eta_cut)
    # so that the post-matching reconstructability filter is still the primary acceptance gate.
    match_min_hits = 0    # truth particle must have >= this many hits to be matchable
    match_min_pt = 0.0    # truth particle must have pT >= this [GeV] to be matchable

    # Default threshold sweep applied to all models unless overridden below.
    default_track_valid_thresholds = [0.5, 0.6, 0.7]
    default_iou_thresholds = [0.5, 0.6, 0.7, 0.8]

    # Per-model threshold overrides. Omitted models use the defaults above.
    # `eval_mode` controls the evaluation-file layout (`old` or `new`).
    # `idx_mode` controls raw parquet event indexing (`paper` or `new`).
    # Either or both of track_valid_thresholds / iou_thresholds can be specified.
    tracking_threshold_overrides = {
        "Paper": {"eval_mode": "old", "idx_mode": "paper", "track_valid_thresholds": [0.5], "iou_thresholds": [0.5]},
        "LCA 900": {"eval_mode": "old", "idx_mode": "paper", "track_valid_thresholds": [0.5], "iou_thresholds": [0.5]},
    }

    # Per-model hit-order handling; "auto" infers the best alignment from file metadata.
    hit_order_by_name = {
        "Paper": "as_saved",
        "Tracking DQ": "as_saved",
        "LCA": "auto",
    }

    # ── Run ───────────────────────────────────────────────────────────────────
    tracking_results, truth_reference_results = load_all_results(
        tracking_fnames=tracking_fnames,
        tracking_configs=tracking_configs,
        default_track_valid_thresholds=default_track_valid_thresholds,
        default_iou_thresholds=default_iou_thresholds,
        tracking_threshold_overrides=tracking_threshold_overrides,
        hit_order_by_name=hit_order_by_name,
        num_events=num_events,
        random_seed=random_seed,
        particle_targets=particle_targets,
        truth_reference_mode=truth_reference_mode,
        match_min_hits=match_min_hits,
        match_min_pt=match_min_pt,
        event_cache_dir=event_cache_dir,
        reuse_saved_event_cache=True,
        write_event_cache=True,
    )

    plot_efficiency(
        tracking_results=tracking_results,
        truth_reference_results=truth_reference_results,
        tracking_fnames=tracking_fnames,
        particle_targets=particle_targets,
        qty_bins=qty_bins,
        qty_symbols=qty_symbols,
        qty_units=qty_units,
        training_colours=training_colours,
        out_dir=out_dir,
        truth_reference_mode=truth_reference_mode,
    )

    print_and_save_summary(
        tracking_results=tracking_results,
        truth_reference_results=truth_reference_results,
        tracking_configs=tracking_configs,
        tracking_fnames=tracking_fnames,
        tracking_threshold_overrides=tracking_threshold_overrides,
        truth_reference_mode=truth_reference_mode,
        out_dir=out_dir,
    )
