"""Simple evaluator for old paper-format TrackML outputs.

This module only supports old-style `__test.h5` files with groups like:

    event_i/
      truth/
      hits/
      parts/
      preds/

It hard-wires the duplicate handling requested for the simplified comparison:

- identical-mask handling: `drop_from_metrics`
- same-majority handling: `postmatch_efficiency`
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _load_track_valid_prob(preds_group):
    if "track_valid_prob" in preds_group:
        return np.asarray(preds_group["track_valid_prob"], dtype=np.float32)
    return None


def _load_track_iou(preds_group):
    if "query_iou" in preds_group:
        return np.asarray(preds_group["query_iou"], dtype=np.float32)
    return None


def _sorted_event_ids(h5_file: h5py.File, num_events: int | None = None) -> list[str]:
    event_ids = sorted(
        (key for key in h5_file.keys() if str(key).startswith("event_")),
        key=lambda key: int(str(key).removeprefix("event_")),
    )
    if num_events is not None:
        return event_ids[:num_events]
    return event_ids


def _event_id_to_event_index(event_id: str, key_mode: str | None = None) -> int:
    event_str = str(event_id)
    if event_str.startswith("event_"):
        event_idx = int(event_str.removeprefix("event_"))
    else:
        event_idx = int(event_str)
    if key_mode == "old":
        event_idx += 29800
    return event_idx


@lru_cache(maxsize=None)
def _event_file_map(test_dir: str, suffix: str) -> dict[int, Path]:
    event_map = {}
    for path in Path(test_dir).glob(f"event*-{suffix}.parquet"):
        event_name = path.stem.removesuffix(f"-{suffix}")
        try:
            event_map[int(event_name.removeprefix("event"))] = path
        except ValueError:
            continue
    return event_map


def _candidate_event_indices(event_id: str, key_mode: str | None = None) -> list[int]:
    base_idx = _event_id_to_event_index(event_id, key_mode=None)
    candidates = [base_idx]
    old_idx = _event_id_to_event_index(event_id, key_mode="old")
    if key_mode == "old":
        candidates = [old_idx, base_idx]
    elif old_idx != base_idx:
        candidates.append(old_idx)
    return candidates


def _resolve_event_path(test_dir: str, suffix: str, event_id: str, key_mode: str | None = None) -> Path:
    event_map = _event_file_map(test_dir, suffix)
    for event_idx in _candidate_event_indices(event_id, key_mode=key_mode):
        if event_idx in event_map:
            return event_map[event_idx]

    available = sorted(event_map)
    available_preview = available[:3]
    if len(available) > 6:
        available_preview = available[:3] + ["..."] + available[-3:]
    elif len(available) > 3:
        available_preview = available

    raise ValueError(
        f"Could not find raw {suffix!r} parquet for {event_id!r} under test_dir={test_dir!r}. "
        f"Tried indices {_candidate_event_indices(event_id, key_mode=key_mode)!r}; "
        f"available examples: {available_preview!r}."
    )


def _prefilter_lookup_event(event_group: h5py.Group, event_id: str, key_mode: str | None) -> tuple[str, str | None]:
    sample_id = event_group.attrs.get("sample_id")
    if sample_id is None:
        return event_id, key_mode
    if isinstance(sample_id, np.ndarray):
        sample_id = sample_id.item()
    return str(int(sample_id)), None


def _load_prefilter_truth_parts(
    *,
    event_id: str,
    data_config: dict,
    key_mode: str | None,
) -> pd.DataFrame:
    test_dir = str(data_config["test_dir"])
    parts_path = _resolve_event_path(test_dir, "parts", event_id, key_mode=key_mode)
    hits_path = _resolve_event_path(test_dir, "hits", event_id, key_mode=key_mode)

    particles = pd.read_parquet(parts_path).copy()
    hits = pd.read_parquet(hits_path, columns=["particle_id", "volume_id"])

    hit_volume_ids = data_config.get("hit_volume_ids")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)]

    particles["particle_pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    particles["particle_phi"] = np.arctan2(particles["py"], particles["px"])
    particle_p = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["particle_eta"] = np.arctanh(particles["pz"] / particle_p)

    min_pt = data_config.get("particle_min_pt")
    if min_pt is not None:
        particles = particles[particles["particle_pt"] > float(min_pt)]

    max_abs_eta = data_config.get("particle_max_abs_eta")
    if max_abs_eta is not None:
        particles = particles[particles["particle_eta"].abs() < float(max_abs_eta)]

    hit_counts = hits["particle_id"].value_counts()
    particles["n_hits_prefilter"] = particles["particle_id"].map(hit_counts).fillna(0).astype(np.int64)

    min_hits = int(data_config.get("particle_min_num_hits", 3))
    particles = particles[particles["n_hits_prefilter"] >= min_hits].copy()

    max_n = data_config.get("event_max_num_particles")
    if max_n is not None and len(particles) > int(max_n):
        if data_config.get("strict_max_objects", False):
            raise ValueError(f"Event {event_id!r} has {len(particles)} particles, but limit is {max_n}.")
        particles = particles.iloc[: int(max_n)].copy()

    particle_vz = particles["vz"].to_numpy(dtype=np.float32) if "vz" in particles.columns else np.full(len(particles), np.nan, dtype=np.float32)
    return pd.DataFrame(
        {
            "particle_id": particles["particle_id"].to_numpy(dtype=np.int64),
            "particle_pt": particles["particle_pt"].to_numpy(dtype=np.float32),
            "particle_eta": particles["particle_eta"].to_numpy(dtype=np.float32),
            "particle_phi": particles["particle_phi"].to_numpy(dtype=np.float32),
            "particle_vz": particle_vz,
            "n_hits_prefilter": particles["n_hits_prefilter"].to_numpy(dtype=np.int64),
            "event_id": event_id,
        }
    )


def _load_prefilter_hits(
    *,
    event_id: str,
    data_config: dict,
    key_mode: str | None,
) -> pd.DataFrame:
    test_dir = str(data_config["test_dir"])
    hits_path = _resolve_event_path(test_dir, "hits", event_id, key_mode=key_mode)
    hits = pd.read_parquet(hits_path, columns=["hit_id", "particle_id", "volume_id", "x", "y"]).copy()

    hit_volume_ids = data_config.get("hit_volume_ids")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)]

    hit_eval_path = data_config.get("hit_eval_test") or data_config.get("hit_eval_path")
    if hit_eval_path:
        sample_key = str(int(event_id))
        with h5py.File(hit_eval_path, "r") as hit_eval_file:
            if f"{sample_key}/preds/final/hit_filter/hit_on_valid_particle_prob" in hit_eval_file:
                hit_filter_probs = hit_eval_file[f"{sample_key}/preds/final/hit_filter/hit_on_valid_particle_prob"][0]
                hit_filter_pred = hit_filter_probs >= float(data_config.get("hit_filter_threshold", 0.1))
            elif f"{sample_key}/preds/final/hit_filter/key_on_valid_particle_prob" in hit_eval_file:
                hit_filter_probs = hit_eval_file[f"{sample_key}/preds/final/hit_filter/key_on_valid_particle_prob"][0]
                hit_filter_pred = hit_filter_probs >= float(data_config.get("hit_filter_threshold", 0.1))
            else:
                hit_filter_pred = hit_eval_file[f"{sample_key}/preds/final/hit_filter/hit_on_valid_particle"][0]
        if len(hit_filter_pred) != len(hits):
            raise ValueError(
                f"Hit filter prediction length mismatch for event {event_id!r}: "
                f"{len(hit_filter_pred)} predictions for {len(hits)} hits."
            )
        hits = hits[hit_filter_pred]

    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    return hits


def _mean_majority_precision(masks: np.ndarray, valid_tracks: np.ndarray, hit_particle_ids: np.ndarray) -> float:
    scores = []
    for track_idx in np.flatnonzero(valid_tracks):
        assigned_pids = hit_particle_ids[masks[track_idx]]
        if assigned_pids.size == 0:
            continue
        _, counts = np.unique(assigned_pids, return_counts=True)
        scores.append(float(counts.max() / max(int(masks[track_idx].sum()), 1)))
    return float(np.mean(scores)) if scores else float("nan")


def _maybe_realign_hit_particle_ids_for_phi_sort(
    *,
    group: h5py.Group,
    event_id: str,
    hit_particle_ids: np.ndarray,
    masks: np.ndarray,
    valid_tracks: np.ndarray,
    prefilter_data_config: dict | None,
    prefilter_key_mode: str | None,
) -> np.ndarray:
    if prefilter_data_config is None or "hids" not in group["hits"]:
        return hit_particle_ids

    try:
        prefilter_event_id, prefilter_lookup_key_mode = _prefilter_lookup_event(group, event_id, prefilter_key_mode)
        raw_hits = _load_prefilter_hits(
            event_id=prefilter_event_id,
            data_config=prefilter_data_config,
            key_mode=prefilter_lookup_key_mode,
        )
    except Exception as exception:
        warnings.warn(f"Could not validate hit-axis ordering for {event_id!r}: {exception}", stacklevel=2)
        return hit_particle_ids

    stored_hit_ids = np.asarray(group["hits"]["hids"], dtype=np.int64)
    raw_hit_ids = raw_hits["hit_id"].to_numpy(dtype=np.int64)
    if len(stored_hit_ids) != len(raw_hit_ids):
        return hit_particle_ids
    if set(stored_hit_ids.tolist()) != set(raw_hit_ids.tolist()):
        return hit_particle_ids

    phi_sort_idx = np.argsort(raw_hits["phi"].to_numpy(dtype=np.float32), kind="stable")
    phi_sorted_hit_ids = raw_hit_ids[phi_sort_idx]
    if np.array_equal(stored_hit_ids, phi_sorted_hit_ids):
        return hit_particle_ids

    stored_idx_by_hit_id = {int(hit_id): idx for idx, hit_id in enumerate(stored_hit_ids)}
    reorder_idx = np.asarray([stored_idx_by_hit_id[int(hit_id)] for hit_id in phi_sorted_hit_ids], dtype=np.int64)

    current_score = _mean_majority_precision(masks, valid_tracks, hit_particle_ids)
    phi_sorted_score = _mean_majority_precision(masks, valid_tracks, hit_particle_ids[reorder_idx])
    if np.isfinite(phi_sorted_score) and (
        not np.isfinite(current_score) or phi_sorted_score > current_score + 0.05
    ):
        warnings.warn(
            f"Detected likely hit-axis sort mismatch for {event_id!r}; "
            f"using phi-sorted hit labels for matching ({current_score:.3f} -> {phi_sorted_score:.3f}).",
            stacklevel=2,
        )
        return hit_particle_ids[reorder_idx]

    return hit_particle_ids


def _shared_hit_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return int(np.count_nonzero(mask_a & mask_b))


def _flag_near_duplicates(
    tracks: pd.DataFrame,
    masks: np.ndarray,
    shared_hits_threshold: int,
    random_seed: int,
) -> np.ndarray:
    near_duplicate = np.zeros(len(tracks), dtype=bool)
    if len(tracks) < 2 or shared_hits_threshold < 0:
        return near_duplicate

    rng = np.random.default_rng(random_seed)
    identical_duplicate = tracks["identical_duplicate"].to_numpy(dtype=bool)
    for (_, majority_pid), group in tracks.groupby(["event_id", "majority_particle_id"], sort=False):
        if int(majority_pid) <= 0:
            continue

        group_idx = group.index.to_numpy(dtype=np.int64)
        candidate_idx = group_idx[~identical_duplicate[group_idx]]
        if candidate_idx.size < 2:
            continue

        adjacency: dict[int, set[int]] = {int(track_idx): set() for track_idx in candidate_idx}
        for idx_pos, track_idx in enumerate(candidate_idx):
            for other_idx in candidate_idx[idx_pos + 1 :]:
                if _shared_hit_count(masks[track_idx], masks[other_idx]) > shared_hits_threshold:
                    adjacency[int(track_idx)].add(int(other_idx))
                    adjacency[int(other_idx)].add(int(track_idx))

        seen: set[int] = set()
        for track_idx in candidate_idx:
            track_idx = int(track_idx)
            if track_idx in seen or not adjacency[track_idx]:
                continue

            component = []
            stack = [track_idx]
            while stack:
                current_idx = stack.pop()
                if current_idx in seen:
                    continue
                seen.add(current_idx)
                component.append(current_idx)
                stack.extend(sorted(adjacency[current_idx] - seen))

            if len(component) <= 1:
                continue

            keep_idx = int(rng.choice(np.asarray(component, dtype=np.int64)))
            for drop_idx in component:
                if drop_idx != keep_idx:
                    near_duplicate[drop_idx] = True

    return near_duplicate


def _resolve_drop_same_hits_within_n(
    drop_same_hits_within_n: int | None,
    drop_one_hit_duplicates: bool,
) -> int | None:
    if drop_same_hits_within_n is not None:
        return int(drop_same_hits_within_n)
    if drop_one_hit_duplicates:
        return 1
    return None


def _load_old_event(
    group: h5py.Group,
    event_id: str,
    eta_cut: float,
    track_valid_threshold: float,
    iou_threshold: float,
    truth_reference_mode: str,
    prefilter_data_config: dict | None = None,
    prefilter_key_mode: str | None = None,
    drop_same_hits_within_n: int | None = None,
    drop_one_hit_duplicates: bool = False,
    duplicate_removal_random_seed: int = 0,
):
    eval_parts = pd.DataFrame({
        "particle_id": np.asarray(group["parts"]["pids"], dtype=np.int64),
        "particle_pt": np.asarray(group["parts"]["pts"], dtype=np.float32),
        "particle_eta": np.asarray(group["parts"]["etas"], dtype=np.float32),
        "particle_phi": np.asarray(group["parts"]["phis"], dtype=np.float32),
        "particle_vz": np.asarray(group["parts"]["vzs"], dtype=np.float32),
    })
    if "n_hits" in group["parts"]:
        eval_parts["n_hits_prefilter"] = np.asarray(group["parts"]["n_hits"], dtype=np.int64)
    eval_parts["event_id"] = event_id

    truth_particle_ids = np.asarray(group["truth"]["particle_id"], dtype=np.int64)
    hit_particle_ids = np.asarray(group["hits"]["pids"], dtype=np.int64)
    mask_values = np.asarray(group["preds"]["masks"])
    masks = mask_values > 0

    n_true_hits_by_pid = pd.Series(truth_particle_ids).value_counts()
    eval_parts["n_true_hits_postfilter"] = eval_parts["particle_id"].map(n_true_hits_by_pid).fillna(0).astype(np.int64)
    if truth_reference_mode == "pre_filter":
        if prefilter_data_config is not None:
            prefilter_event_id, prefilter_lookup_key_mode = _prefilter_lookup_event(group, event_id, prefilter_key_mode)
            parts = _load_prefilter_truth_parts(
                event_id=prefilter_event_id,
                data_config=prefilter_data_config,
                key_mode=prefilter_lookup_key_mode,
            )
            parts["event_id"] = event_id
        elif "n_hits_prefilter" in eval_parts.columns:
            parts = eval_parts.copy()
        else:
            raise ValueError(
                f"truth_reference_mode='pre_filter' requested for {event_id!r}, but parts/n_hits is missing from the eval file "
                "and no prefilter_data_config was provided."
            )
        parts["n_true_hits"] = parts["n_hits_prefilter"]
    else:
        parts = eval_parts.copy()
        parts["n_true_hits"] = parts["n_true_hits_postfilter"]
    parts["n_true_hits_postfilter"] = parts["particle_id"].map(n_true_hits_by_pid).fillna(0).astype(np.int64)
    parts["valid"] = parts["n_true_hits"] >= 3
    parts["reconstructable_no_pt"] = parts["valid"] & (parts["particle_eta"].abs() < eta_cut)
    reference_hits_by_pid = dict(
        zip(
            parts["particle_id"].astype(np.int64),
            parts["n_true_hits"].astype(np.int64),
            strict=False,
        )
    )

    class_scores = np.asarray(group["preds"]["class_preds"])
    track_valid_prob = _load_track_valid_prob(group["preds"])
    track_iou = _load_track_iou(group["preds"])

    if track_valid_prob is None:
        class_pred = class_scores.argmax(-1) == 0
    else:
        class_pred = np.isfinite(track_valid_prob) & (track_valid_prob >= track_valid_threshold)

    if iou_threshold > 0:
        if track_iou is None:
            raise ValueError(
                f"iou_threshold={iou_threshold} requested for {event_id!r}, but no IoU predictions were found."
            )
        class_pred = class_pred & np.isfinite(track_iou) & (track_iou >= iou_threshold)

    n_pred_hits = masks.sum(axis=-1)
    valid_tracks = class_pred & (n_pred_hits >= 3)
    hit_particle_ids = _maybe_realign_hit_particle_ids_for_phi_sort(
        group=group,
        event_id=event_id,
        hit_particle_ids=hit_particle_ids,
        masks=masks,
        valid_tracks=valid_tracks,
        prefilter_data_config=prefilter_data_config,
        prefilter_key_mode=prefilter_key_mode,
    )

    tracks = pd.DataFrame({
        "event_id": event_id,
        "n_pred_hits": n_pred_hits.astype(np.int64),
        "valid": valid_tracks,
        "identical_duplicate": np.zeros(len(masks), dtype=bool),
        "majority_particle_id": np.full(len(masks), -1, dtype=np.int64),
        "n_true_hits": np.full(len(masks), -1, dtype=np.int64),
        "n_matched_hits": np.full(len(masks), -1, dtype=np.int64),
        "precision_raw": np.full(len(masks), -1.0, dtype=np.float32),
        "recall_raw": np.full(len(masks), -1.0, dtype=np.float32),
        "eff_dm_raw": np.zeros(len(masks), dtype=bool),
        "eff_perfect_raw": np.zeros(len(masks), dtype=bool),
        "eff_lhc_raw": np.zeros(len(masks), dtype=bool),
    })
    if track_valid_prob is not None:
        tracks["track_valid_prob"] = track_valid_prob
    if track_iou is not None:
        tracks["track_iou"] = track_iou

    seen_masks: set[bytes] = set()
    for track_idx in range(len(masks)):
        if not valid_tracks[track_idx]:
            continue

        mask = masks[track_idx]
        mask_bytes = mask.tobytes()
        if mask_bytes in seen_masks:
            tracks.loc[track_idx, "identical_duplicate"] = True
        else:
            seen_masks.add(mask_bytes)

        assigned_pids = hit_particle_ids[mask]
        if assigned_pids.size == 0:
            continue

        majority_pids, majority_counts = np.unique(assigned_pids, return_counts=True)
        best_idx = int(np.argmax(majority_counts))
        majority_pid = int(majority_pids[best_idx])
        n_matched_hits = int(majority_counts[best_idx])
        n_true_hits = int(reference_hits_by_pid.get(majority_pid, np.sum(hit_particle_ids == majority_pid)))

        precision = n_matched_hits / max(int(mask.sum()), 1)
        recall = n_matched_hits / max(n_true_hits, 1)

        tracks.loc[track_idx, "majority_particle_id"] = majority_pid
        tracks.loc[track_idx, "n_true_hits"] = n_true_hits
        tracks.loc[track_idx, "n_matched_hits"] = n_matched_hits
        tracks.loc[track_idx, "precision_raw"] = precision
        tracks.loc[track_idx, "recall_raw"] = recall
        tracks.loc[track_idx, "eff_dm_raw"] = (precision > 0.5) and (recall > 0.5)
        tracks.loc[track_idx, "eff_perfect_raw"] = (precision == 1.0) and (recall == 1.0)
        tracks.loc[track_idx, "eff_lhc_raw"] = precision > 0.75

    tracks = tracks.loc[tracks["valid"]].reset_index(drop=True)
    parts = _apply_fixed_duplicate_modes(
        tracks,
        parts,
        masks=masks[valid_tracks],
        drop_same_hits_within_n=_resolve_drop_same_hits_within_n(
            drop_same_hits_within_n,
            drop_one_hit_duplicates,
        ),
        duplicate_removal_random_seed=duplicate_removal_random_seed,
    )
    return tracks, parts


def _apply_fixed_duplicate_modes(
    tracks: pd.DataFrame,
    parts: pd.DataFrame,
    *,
    masks: np.ndarray,
    drop_same_hits_within_n: int | None = None,
    duplicate_removal_random_seed: int = 0,
) -> pd.DataFrame:
    """Apply the fixed duplicate-handling policy for the simple old evaluator."""
    near_duplicate = (
        _flag_near_duplicates(
            tracks,
            masks,
            shared_hits_threshold=int(drop_same_hits_within_n),
            random_seed=duplicate_removal_random_seed,
        )
        if drop_same_hits_within_n is not None and int(drop_same_hits_within_n) > 0
        else np.zeros(len(tracks), dtype=bool)
    )
    included = ~tracks["identical_duplicate"].to_numpy(dtype=bool) & ~near_duplicate
    metric_eff_dm = tracks["eff_dm_raw"].to_numpy(dtype=bool).copy()
    metric_eff_perfect = tracks["eff_perfect_raw"].to_numpy(dtype=bool).copy()
    metric_eff_lhc = tracks["eff_lhc_raw"].to_numpy(dtype=bool).copy()
    metric_duplicate_dm = np.zeros(len(tracks), dtype=bool)
    metric_duplicate_perfect = np.zeros(len(tracks), dtype=bool)
    metric_duplicate_lhc = np.zeros(len(tracks), dtype=bool)

    for (_, majority_pid), group in tracks.groupby(["event_id", "majority_particle_id"], sort=False):
        if int(majority_pid) <= 0:
            continue
        group_idx_all = group.index.to_numpy(dtype=np.int64)
        group_idx = group_idx_all[included[group_idx_all]]
        if group_idx.size == 0:
            continue

        dm_candidates = group_idx[metric_eff_dm[group_idx]]
        if dm_candidates.size > 0:
            keep = dm_candidates[0]
            metric_duplicate_dm[group_idx[group_idx != keep]] = True
            metric_eff_dm[group_idx] = False
            metric_eff_dm[keep] = True

        perfect_candidates = group_idx[metric_eff_perfect[group_idx]]
        if perfect_candidates.size > 0:
            keep = perfect_candidates[0]
            metric_duplicate_perfect[group_idx[group_idx != keep]] = True
            metric_eff_perfect[group_idx] = False
            metric_eff_perfect[keep] = True

        lhc_candidates = group_idx[metric_eff_lhc[group_idx]]
        if lhc_candidates.size > 0:
            keep = lhc_candidates[0]
            metric_duplicate_lhc[group_idx[group_idx != keep]] = True
            metric_eff_lhc[group_idx] = False
            metric_eff_lhc[keep] = True

    metric_eff_dm[~included] = False
    metric_eff_perfect[~included] = False
    metric_eff_lhc[~included] = False

    tracks["metric_included"] = included
    tracks["near_duplicate"] = near_duplicate
    tracks["metric_eff_dm"] = metric_eff_dm
    tracks["metric_eff_perfect"] = metric_eff_perfect
    tracks["metric_eff_lhc"] = metric_eff_lhc
    tracks["metric_duplicate_dm"] = metric_duplicate_dm
    tracks["metric_duplicate_perfect"] = metric_duplicate_perfect
    tracks["metric_duplicate_lhc"] = metric_duplicate_lhc

    part_keys = pd.MultiIndex.from_arrays([parts["event_id"], parts["particle_id"]])
    dm_track_keys = pd.MultiIndex.from_frame(
        tracks.loc[tracks["metric_eff_dm"] & tracks["metric_included"], ["event_id", "majority_particle_id"]].rename(
            columns={"majority_particle_id": "particle_id"}
        )
    )
    perfect_track_keys = pd.MultiIndex.from_frame(
        tracks.loc[tracks["metric_eff_perfect"] & tracks["metric_included"], ["event_id", "majority_particle_id"]].rename(
            columns={"majority_particle_id": "particle_id"}
        )
    )
    lhc_track_keys = pd.MultiIndex.from_frame(
        tracks.loc[tracks["metric_eff_lhc"] & tracks["metric_included"], ["event_id", "majority_particle_id"]].rename(
            columns={"majority_particle_id": "particle_id"}
        )
    )

    parts["metric_eff_dm"] = part_keys.isin(dm_track_keys)
    parts["metric_eff_perfect"] = part_keys.isin(perfect_track_keys)
    parts["metric_eff_lhc"] = part_keys.isin(lhc_track_keys)
    return parts


def evaluate_file(
    fname: str,
    num_events: int | None = None,
    eta_cut: float = 4.0,
    pt_cut: float = 1.0,
    track_valid_threshold: float = 0.5,
    iou_threshold: float = 0.0,
    truth_reference_mode: str = "post_filter",
    prefilter_data_config: dict | None = None,
    prefilter_key_mode: str | None = None,
    drop_same_hits_within_n: int | None = None,
    drop_one_hit_duplicates: bool = False,
    duplicate_removal_random_seed: int = 0,
):
    """Evaluate an old-format paper file with fixed duplicate handling."""
    if truth_reference_mode not in {"post_filter", "pre_filter"}:
        raise ValueError(f"Unknown truth_reference_mode={truth_reference_mode!r}.")
    tracks_list = []
    parts_list = []

    with h5py.File(fname, "r") as h5_file:
        for event_id in _sorted_event_ids(h5_file, num_events=num_events):
            tracks, parts = _load_old_event(
                h5_file[event_id],
                event_id=event_id,
                eta_cut=eta_cut,
                track_valid_threshold=track_valid_threshold,
                iou_threshold=iou_threshold,
                truth_reference_mode=truth_reference_mode,
                prefilter_data_config=prefilter_data_config,
                prefilter_key_mode=prefilter_key_mode,
                drop_same_hits_within_n=drop_same_hits_within_n,
                drop_one_hit_duplicates=drop_one_hit_duplicates,
                duplicate_removal_random_seed=duplicate_removal_random_seed,
            )
            parts["reconstructable"] = parts["reconstructable_no_pt"] & (parts["particle_pt"] > pt_cut)
            tracks_list.append(tracks)
            parts_list.append(parts)

    if tracks_list:
        tracks = pd.concat(tracks_list, ignore_index=True)
    else:
        tracks = pd.DataFrame()

    if parts_list:
        parts = pd.concat(parts_list, ignore_index=True)
    else:
        parts = pd.DataFrame()

    return tracks, parts


@dataclass(frozen=True)
class OldEvalSummary:
    n_events: int
    n_particles: int
    n_valid_particles: int
    n_reconstructable_particles: int
    n_valid_tracks: int
    n_included_tracks: int
    dm_efficiency: float
    perfect_efficiency: float
    fake_rate: float
    identical_duplicate_rate: float
    near_duplicate_rate: float
    excluded_identical_duplicate_rate: float
    excluded_near_duplicate_rate: float
    same_majority_duplicate_rate_dm: float


def summarize_results(tracks: pd.DataFrame, parts: pd.DataFrame) -> OldEvalSummary:
    """Summarise the fixed-mode old-format evaluation."""
    n_events = int(parts["event_id"].nunique()) if not parts.empty else 0
    n_particles = len(parts)
    n_valid_particles = int(parts["valid"].sum()) if "valid" in parts else 0
    reconstructable = parts["reconstructable"].to_numpy(dtype=bool) if "reconstructable" in parts else np.array([], dtype=bool)
    included = tracks["metric_included"].to_numpy(dtype=bool) if "metric_included" in tracks else np.array([], dtype=bool)

    dm_efficiency = float(parts.loc[reconstructable, "metric_eff_dm"].mean()) if reconstructable.any() else float("nan")
    perfect_efficiency = (
        float(parts.loc[reconstructable, "metric_eff_perfect"].mean()) if reconstructable.any() else float("nan")
    )
    fake_rate = float((~tracks.loc[included, "metric_eff_dm"]).mean()) if included.any() else float("nan")

    return OldEvalSummary(
        n_events=n_events,
        n_particles=n_particles,
        n_valid_particles=n_valid_particles,
        n_reconstructable_particles=int(reconstructable.sum()),
        n_valid_tracks=len(tracks),
        n_included_tracks=int(included.sum()),
        dm_efficiency=dm_efficiency,
        perfect_efficiency=perfect_efficiency,
        fake_rate=fake_rate,
        identical_duplicate_rate=float(tracks["identical_duplicate"].mean()) if len(tracks) else float("nan"),
        near_duplicate_rate=float(tracks["near_duplicate"].mean()) if "near_duplicate" in tracks and len(tracks) else float("nan"),
        excluded_identical_duplicate_rate=float(tracks["identical_duplicate"].mean()) if len(tracks) else float("nan"),
        excluded_near_duplicate_rate=float(tracks["near_duplicate"].mean()) if "near_duplicate" in tracks and len(tracks) else float("nan"),
        same_majority_duplicate_rate_dm=float(tracks["metric_duplicate_dm"].mean()) if len(tracks) else float("nan"),
    )


def format_summary(
    summary: OldEvalSummary,
    *,
    fname: str,
    eta_cut: float,
    pt_cut: float,
    track_valid_threshold: float,
    iou_threshold: float,
    drop_same_hits_within_n: int | None = None,
) -> str:
    """Format an evaluation summary for console output."""
    return "\n".join(
        [
            "---------------------------",
            fname,
            f"eta cut: {eta_cut}",
            f"pt cut: {pt_cut}",
            f"track valid threshold: {track_valid_threshold}",
            f"iou threshold: {iou_threshold}",
            f"drop same-particle tracks sharing more than N hits: {drop_same_hits_within_n}",
            f"N events: {summary.n_events}, N particles: {summary.n_particles}, N valid tracks: {summary.n_valid_tracks}",
            f"N reconstructable particles: {summary.n_reconstructable_particles}",
            f"N included tracks: {summary.n_included_tracks}",
            f"DM efficiency: {summary.dm_efficiency:.1%}",
            f"Perfect efficiency: {summary.perfect_efficiency:.1%}",
            f"Fake rate: {summary.fake_rate:.1%}",
            f"Identical duplicate rate: {summary.identical_duplicate_rate:.1%}",
            f"Excluded identical duplicate rate: {summary.excluded_identical_duplicate_rate:.1%}",
            f"Near-duplicate rate: {summary.near_duplicate_rate:.1%}",
            f"Excluded near-duplicate rate: {summary.excluded_near_duplicate_rate:.1%}",
            f"Same-majority duplicate rate (DM): {summary.same_majority_duplicate_rate_dm:.1%}",
        ]
    )
