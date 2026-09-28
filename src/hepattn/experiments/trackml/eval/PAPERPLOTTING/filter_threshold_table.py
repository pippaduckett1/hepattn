# #!/usr/bin/env python3
# """Summarize filter-eval hit counts and reconstructable efficiency at one threshold."""

# import argparse
# import pathlib

# import h5py
# import numpy as np
# from tqdm import tqdm


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Report Hits (Pre), Hits (Post), and reconstructable efficiency for one filter-eval threshold.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("eval_file", type=pathlib.Path, help="Path to the filter evaluation HDF5 file.")
#     parser.add_argument("--threshold", type=float, required=True, help="Score threshold used to keep hits.")
#     parser.add_argument(
#         "--reconstructable-min-hits",
#         type=int,
#         default=3,
#         help="Minimum kept hits required for a valid particle to count as reconstructable after filtering.",
#     )
#     parser.add_argument("--max-events", type=int, default=None, help="Optional cap on the number of events to scan.")
#     parser.add_argument("--task-name", type=str, default="hit_filter", help="Task group under preds/final.")
#     parser.add_argument(
#         "--pred-dataset",
#         type=str,
#         default="hit_on_valid_particle_prob",
#         help="Prediction dataset inside preds/final/<task-name>/.",
#     )
#     return parser.parse_args()


# def _sorted_event_keys(file_handle: h5py.File) -> list[str]:
#     def _key(key: str):
#         return (0, int(key)) if key.isdigit() else (1, key)

#     return sorted(file_handle.keys(), key=_key)


# def summarize_threshold(
#     eval_path: pathlib.Path,
#     threshold: float,
#     reconstructable_min_hits: int,
#     max_events: int | None,
#     task_name: str,
#     pred_dataset: str,
# ) -> dict:
#     hits_pre = 0
#     hits_post = 0
#     valid_particles = 0
#     reconstructable_particles = 0
#     nonfinite_scores = 0

#     with h5py.File(eval_path, "r") as f:
#         keys = _sorted_event_keys(f)
#         if max_events is not None:
#             keys = keys[:max_events]

#         for key in tqdm(keys, desc=f"Scanning events ({eval_path.name})"):
#             score = np.asarray(f[f"{key}/preds/final/{task_name}/{pred_dataset}"])[0].astype(np.float64, copy=False)
#             particle_hit_valid = np.asarray(f[f"{key}/targets/particle_hit_valid"])[0].astype(bool)
#             particle_valid = np.asarray(f[f"{key}/targets/particle_valid"])[0].astype(bool)

#             finite = np.isfinite(score)
#             nonfinite_scores += int(score.size - np.sum(finite))
#             if not np.any(finite):
#                 continue

#             score = score[finite]
#             particle_hit_valid = particle_hit_valid[:, finite]
#             kept_hits = score >= threshold

#             hits_pre += int(score.size)
#             hits_post += int(np.sum(kept_hits))

#             pred_hits_per_particle = particle_hit_valid.astype(np.int32) @ kept_hits.astype(np.int32)
#             valid_particles += int(np.sum(particle_valid))
#             reconstructable_particles += int(np.sum((pred_hits_per_particle >= reconstructable_min_hits) & particle_valid))

#     reconstructable_efficiency = reconstructable_particles / valid_particles if valid_particles else 0.0
#     return {
#         "threshold": float(threshold),
#         "hits_pre": int(hits_pre),
#         "hits_post": int(hits_post),
#         "valid_particles": int(valid_particles),
#         "reconstructable_particles": int(reconstructable_particles),
#         "reconstructable_efficiency": float(reconstructable_efficiency),
#         "nonfinite_scores": int(nonfinite_scores),
#     }


# def main():
#     args = parse_args()
#     summary = summarize_threshold(
#         eval_path=args.eval_file,
#         threshold=args.threshold,
#         reconstructable_min_hits=args.reconstructable_min_hits,
#         max_events=args.max_events,
#         task_name=args.task_name,
#         pred_dataset=args.pred_dataset,
#     )

#     print(f"Eval file: {args.eval_file}")
#     print(f"Threshold: {summary['threshold']:.4f}")
#     print(f"Hits (Pre): {summary['hits_pre']}")
#     print(f"Hits (Post): {summary['hits_post']}")
#     print(
#         f"Reconstructable Efficiency (>= {args.reconstructable_min_hits} kept hits): "
#         f"{summary['reconstructable_efficiency']:.6f}"
#     )
#     print(f"Valid particles: {summary['valid_particles']}")
#     print(f"Reconstructable particles after filtering: {summary['reconstructable_particles']}")
#     if summary["nonfinite_scores"]:
#         print(f"Warning: ignored {summary['nonfinite_scores']} non-finite scores.")

#     print("\nLaTeX row:")
#     print("Hits (Pre) & Hits (Post) & Reconstructable Efficiency \\\\")
#     print(
#         f"{summary['hits_pre']} & {summary['hits_post']} & "
#         f"{summary['reconstructable_efficiency']:.4f} \\\\"
#     )


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""Summarize filter-eval hit counts and reconstructable efficiency at one threshold."""

import argparse
import math
import pathlib

import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Report Hits (Pre), Hits (Post), and reconstructable efficiency for a filter-eval threshold. "
            "If --threshold is omitted, select the highest threshold that keeps the target reconstructable efficiency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("eval_file", type=pathlib.Path, help="Path to the filter evaluation HDF5 file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Score threshold used to keep hits. If omitted, derive it from --target-reconstructable-efficiency.",
    )
    parser.add_argument(
        "--target-reconstructable-efficiency",
        type=float,
        default=0.99,
        help="Efficiency target used to choose a threshold when --threshold is omitted.",
    )
    parser.add_argument(
        "--reconstructable-min-hits",
        type=int,
        default=3,
        help="Minimum kept hits required for a valid particle to count as reconstructable after filtering.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on the number of events to scan.")
    parser.add_argument("--task-name", type=str, default="hit_filter", help="Task group under preds/final.")
    parser.add_argument(
        "--pred-dataset",
        type=str,
        default="hit_on_valid_particle_prob",
        help="Prediction dataset inside preds/final/<task-name>/.",
    )
    args = parser.parse_args()
    if args.threshold is not None and not math.isfinite(args.threshold):
        parser.error("--threshold must be finite.")
    if not 0.0 < args.target_reconstructable_efficiency <= 1.0:
        parser.error("--target-reconstructable-efficiency must be in the range (0, 1].")
    if args.reconstructable_min_hits < 1:
        parser.error("--reconstructable-min-hits must be at least 1.")
    return args


def _sorted_event_keys(file_handle: h5py.File) -> list[str]:
    def _key(key: str):
        return (0, int(key)) if key.isdigit() else (1, key)

    return sorted(file_handle.keys(), key=_key)


def find_threshold_for_reconstructable_efficiency(
    eval_path: pathlib.Path,
    target_efficiency: float,
    reconstructable_min_hits: int,
    max_events: int | None,
    task_name: str,
    pred_dataset: str,
) -> dict:
    particle_thresholds = []
    valid_particles = 0
    possible_particles = 0

    with h5py.File(eval_path, "r") as f:
        keys = _sorted_event_keys(f)
        if max_events is not None:
            keys = keys[:max_events]

        for key in tqdm(keys, desc=f"Selecting threshold ({eval_path.name})"):
            score = np.asarray(f[f"{key}/preds/final/{task_name}/{pred_dataset}"])[0].astype(np.float64, copy=False)
            particle_hit_valid = np.asarray(f[f"{key}/targets/particle_hit_valid"])[0].astype(bool)
            particle_valid = np.asarray(f[f"{key}/targets/particle_valid"])[0].astype(bool)

            finite = np.isfinite(score)
            if not np.any(finite):
                continue

            score = score[finite]
            particle_hit_valid = particle_hit_valid[:, finite]

            valid_particle_indices = np.flatnonzero(particle_valid)
            valid_particles += int(valid_particle_indices.size)

            for particle_idx in valid_particle_indices:
                particle_scores = score[particle_hit_valid[particle_idx]]
                if particle_scores.size < reconstructable_min_hits:
                    continue
                threshold = np.partition(particle_scores, -reconstructable_min_hits)[-reconstructable_min_hits]
                particle_thresholds.append(float(threshold))
                possible_particles += 1

    if valid_particles == 0:
        raise ValueError("No valid particles found while selecting the threshold.")

    max_efficiency = possible_particles / valid_particles
    if max_efficiency < target_efficiency:
        raise ValueError(
            "Target reconstructable efficiency is unattainable: "
            f"maximum is {max_efficiency:.6f} from {possible_particles}/{valid_particles} valid particles."
        )

    required_particles = int(math.ceil(target_efficiency * valid_particles))
    thresholds_desc = np.sort(np.asarray(particle_thresholds, dtype=np.float64))[::-1]
    threshold = float(thresholds_desc[required_particles - 1])

    return {
        "threshold": threshold,
        "target_reconstructable_efficiency": float(target_efficiency),
        "valid_particles": int(valid_particles),
        "required_reconstructable_particles": int(required_particles),
        "max_reconstructable_efficiency": float(max_efficiency),
    }


def summarize_threshold(
    eval_path: pathlib.Path,
    threshold: float,
    reconstructable_min_hits: int,
    max_events: int | None,
    task_name: str,
    pred_dataset: str,
) -> dict:
    hits_pre = 0
    hits_post = 0
    valid_particles = 0
    reconstructable_particles = 0
    nonfinite_scores = 0

    with h5py.File(eval_path, "r") as f:
        keys = _sorted_event_keys(f)
        if max_events is not None:
            keys = keys[:max_events]

        for key in tqdm(keys, desc=f"Scanning events ({eval_path.name})"):
            score = np.asarray(f[f"{key}/preds/final/{task_name}/{pred_dataset}"])[0].astype(np.float64, copy=False)
            particle_hit_valid = np.asarray(f[f"{key}/targets/particle_hit_valid"])[0].astype(bool)
            particle_valid = np.asarray(f[f"{key}/targets/particle_valid"])[0].astype(bool)

            finite = np.isfinite(score)
            nonfinite_scores += int(score.size - np.sum(finite))
            if not np.any(finite):
                continue

            score = score[finite]
            particle_hit_valid = particle_hit_valid[:, finite]
            kept_hits = score >= threshold

            hits_pre += int(score.size)
            hits_post += int(np.sum(kept_hits))

            pred_hits_per_particle = particle_hit_valid.astype(np.int32) @ kept_hits.astype(np.int32)
            valid_particles += int(np.sum(particle_valid))
            reconstructable_particles += int(np.sum((pred_hits_per_particle >= reconstructable_min_hits) & particle_valid))

    reconstructable_efficiency = reconstructable_particles / valid_particles if valid_particles else 0.0
    return {
        "threshold": float(threshold),
        "hits_pre": int(hits_pre),
        "hits_post": int(hits_post),
        "valid_particles": int(valid_particles),
        "reconstructable_particles": int(reconstructable_particles),
        "reconstructable_efficiency": float(reconstructable_efficiency),
        "nonfinite_scores": int(nonfinite_scores),
    }


def main():
    args = parse_args()
    selected = None
    threshold = args.threshold
    if threshold is None:
        selected = find_threshold_for_reconstructable_efficiency(
            eval_path=args.eval_file,
            target_efficiency=args.target_reconstructable_efficiency,
            reconstructable_min_hits=args.reconstructable_min_hits,
            max_events=args.max_events,
            task_name=args.task_name,
            pred_dataset=args.pred_dataset,
        )
        threshold = selected["threshold"]

    summary = summarize_threshold(
        eval_path=args.eval_file,
        threshold=threshold,
        reconstructable_min_hits=args.reconstructable_min_hits,
        max_events=args.max_events,
        task_name=args.task_name,
        pred_dataset=args.pred_dataset,
    )

    print(f"Eval file: {args.eval_file}")
    if selected is not None:
        print(
            "Selected threshold for target reconstructable efficiency "
            f"{selected['target_reconstructable_efficiency']:.6f}"
        )
    print(f"Threshold: {summary['threshold']:.4f}")
    print(f"Hits (Pre): {summary['hits_pre']}")
    print(f"Hits (Post): {summary['hits_post']}")
    print(
        f"Reconstructable Efficiency (>= {args.reconstructable_min_hits} kept hits): "
        f"{summary['reconstructable_efficiency']:.6f}"
    )
    print(f"Valid particles: {summary['valid_particles']}")
    print(f"Reconstructable particles after filtering: {summary['reconstructable_particles']}")
    if summary["nonfinite_scores"]:
        print(f"Warning: ignored {summary['nonfinite_scores']} non-finite scores.")

    print("\nLaTeX row:")
    print("Hits (Pre) & Hits (Post) & Reconstructable Efficiency \\\\")
    print(
        f"{summary['hits_pre']} & {summary['hits_post']} & "
        f"{summary['reconstructable_efficiency']:.4f} \\\\"
    )


if __name__ == "__main__":
    main()
