import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _resolve_base_name(timing_dir: Path, model_name: str | None) -> str:
    if model_name is not None:
        return model_name

    times_paths = sorted(timing_dir.glob("*_times.npy"))
    if len(times_paths) != 1:
        raise ValueError(
            f"Expected exactly one '*_times.npy' file in {timing_dir}, found {len(times_paths)}. "
            "Pass --model-name to disambiguate."
        )
    return times_paths[0].name.removesuffix("_times.npy")


def _load_hit_counts(dims_path: Path) -> np.ndarray:
    hit_counts = np.asarray(np.load(dims_path), dtype=np.int64)
    if hit_counts.ndim != 1:
        raise ValueError(f"Expected 1D hit-count array in {dims_path}, found shape {hit_counts.shape}.")
    if len(hit_counts) < 2:
        raise ValueError(f"Need at least 2 events in {dims_path} to make a fit, found {len(hit_counts)}.")
    if len(np.unique(hit_counts)) < 2:
        raise ValueError(f"Need at least 2 distinct hit counts in {dims_path} to fit a line.")
    return hit_counts


def _load_query_counts(query_counts_path: Path) -> np.ndarray:
    query_counts = np.asarray(np.load(query_counts_path), dtype=np.int64)
    if query_counts.ndim != 1:
        raise ValueError(f"Expected 1D query-count array in {query_counts_path}, found shape {query_counts.shape}.")
    return query_counts


def _load_memory_trace_mb(memory_path: Path) -> np.ndarray | None:
    if not memory_path.exists():
        return None

    memory_values = np.asarray(np.load(memory_path), dtype=np.float64) / (1024**2)
    if memory_values.ndim != 1:
        raise ValueError(f"Expected 1D memory array in {memory_path}, found shape {memory_values.shape}.")
    return memory_values


def _resolve_memory_trace_path(memory_dir: Path, model_name: str, metric: str) -> Path | None:
    trace_path = memory_dir / f"{model_name}_test_peak_{metric}_bytes.npy"
    if trace_path.exists():
        return trace_path

    candidates = sorted(memory_dir.glob(f"{model_name}*_test_peak_{metric}_bytes.npy"))
    if not candidates:
        return None
    return candidates[0]


def _resolve_memory_summary_path(memory_dir: Path, model_name: str) -> Path | None:
    summary_path = memory_dir / f"{model_name}_test_memory_summary.json"
    if summary_path.exists():
        return summary_path

    candidates = sorted(memory_dir.glob(f"{model_name}*_test_memory_summary.json"))
    if not candidates:
        return None
    return candidates[0]


def _load_summary_peak_mb(memory_dir: Path, model_name: str, metric: str) -> float | None:
    summary_path = _resolve_memory_summary_path(memory_dir, model_name)
    if summary_path is None:
        return None

    with summary_path.open() as f:
        summary = json.load(f)

    metric_key = f"global_peak_{metric}_gb"
    peak_gb = summary.get(metric_key)
    if peak_gb is None:
        raise ValueError(f"Missing '{metric_key}' in {summary_path}.")
    return float(peak_gb) * 1024.0


def _align_memory_trace(
    memory_values: np.ndarray | None,
    expected_length: int,
    memory_path: Path,
) -> np.ndarray | None:
    if memory_values is None:
        return None
    if len(memory_values) == expected_length:
        return memory_values
    if len(memory_values) > expected_length:
        # MemoryStats records all test forwards, while timing arrays drop warm-start events.
        return memory_values[-expected_length:]
    raise ValueError(
        f"Memory trace in {memory_path} has length {len(memory_values)}, expected at least {expected_length}."
    )


def _expand_optional_list(values: list[str] | None, count: int, arg_name: str) -> list[str | None]:
    if values is None or len(values) == 0:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"{arg_name} expects either 0 values or exactly {count} values, found {len(values)}.")
    return values


def _apply_paper_style() -> list[str]:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    return [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
    ]


def _style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(zorder=0, alpha=0.22, linestyle="--", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _make_model_handles(series: list[dict], colours: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=colours[idx % len(colours)], linewidth=2.4, label=run["label"])
        for idx, run in enumerate(series)
    ]


def _make_memory_encoding_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor="#444444",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label="Allocated GPU memory",
        ),
        Line2D([0], [0], color="#444444", linewidth=2.4, label="Reserved GPU memory"),
    ]


def _resolve_output_path(output: Path | None, series: list[dict]) -> Path:
    if output is None:
        if len(series) == 1:
            return series[0]["timing_dir"] / f'{series[0]["model_name"]}_time_vs_hits.png'
        return Path.cwd() / "trackml_multi_model_time_vs_hits.png"

    if output.suffix:
        return output

    output.mkdir(parents=True, exist_ok=True)
    stem = "trackml_inference_time_vs_hits"
    if len(series) == 1:
        stem = f'{series[0]["model_name"]}_time_vs_hits'
    elif len(series) > 1:
        stem = "trackml_multi_model_time_vs_hits"
    return output / f"{stem}.png"


def _make_related_output_path(base_path: Path, target: str) -> Path:
    stem = base_path.stem
    if "time_vs_hits" in stem:
        stem = stem.replace("time_vs_hits", f"{target}_vs_hits")
    else:
        stem = f"{stem}_{target}_vs_hits"
    return base_path.with_name(f"{stem}{base_path.suffix}")


def _fit_trend_line(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 2 or len(np.unique(xs)) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, deg=1)
    fit_x = np.linspace(xs.min(), xs.max(), 200)
    fit_y = slope * fit_x + intercept
    return fit_x, fit_y


# def main() -> None:
#     colours = _apply_paper_style()
#     timing_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/")
#     memory_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/")
#     # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-qual")]
#     # labels = ["LCA-900"]
#     # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual")]
#     # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30")]
#     model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual")]
#     # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual")]
#     # labels = ["MA-900"]
#     # labels = ["LCA-900"]
#     labels = ["MA-900", "LCA-900"]
#     # labels = ["MA-900", "LCA-900", "LCA-900-w32"]
#     output = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/inference-time-lca-900/")

# def main() -> None:
#     colours = _apply_paper_style()
#     timing_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/")
#     memory_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/")
#     # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-qual")]
#     # labels = ["LCA-900"]
#     # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual")]
#     # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30")]
#     model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual")]
#     # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual")]
#     # labels = ["MA-900"]
#     # labels = ["LCA-900"]
#     labels = ["BS16"]
#     # labels = ["MA-900", "LCA-900", "LCA-900-w32"]
#     output = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/inference-time-lca-bs-16/")

# Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-qual-bs128")


def main() -> None:
    colours = _apply_paper_style()
    timing_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/")
    memory_dir = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/")
    # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-qual")]
    # labels = ["LCA-900"]
    # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual")]
    # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30")]
    model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual")]
    # model_names = [Path("TRK-v8-eta4-lca-eta4-900-th0p25-w64-new-qual")]
    # labels = ["MA-900"]
    # labels = ["LCA-900"]
    labels = ["MA-900", "LCA-900"]
    output = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/inference-time-lca-900-/")

    print("model names: ", model_names)

    series = []
    any_memory = False
    for model_name_override, label_override in zip(model_names, labels, strict=True):
        model_name = _resolve_base_name(timing_dir, model_name_override)
        label = label_override or model_name

        times_path = timing_dir / f"{model_name}_times.npy"
        dims_path = timing_dir / f"{model_name}_dims.npy"
        query_counts_path = timing_dir / f"{model_name}_query_counts.npy"
        peak_alloc_path = _resolve_memory_trace_path(memory_dir, model_name, metric="allocated")
        peak_reserved_path = _resolve_memory_trace_path(memory_dir, model_name, metric="reserved")

        times = np.asarray(np.load(times_path), dtype=np.float64)
        hit_counts = _load_hit_counts(dims_path)
        if len(times) != len(hit_counts):
            raise ValueError(f"Length mismatch: {len(times)} times in {times_path} vs {len(hit_counts)} hit counts in {dims_path}.")

        slope, intercept = np.polyfit(hit_counts, times, deg=1)
        fit_x = np.linspace(hit_counts.min(), hit_counts.max(), 200)
        fit_y = slope * fit_x + intercept

        query_counts = None
        query_slope = None
        query_intercept = None
        query_fit_x = None
        query_fit_y = None
        if query_counts_path.exists():
            query_counts = _load_query_counts(query_counts_path)
            if len(query_counts) != len(times):
                raise ValueError(f"Length mismatch: {len(times)} times in {times_path} vs {len(query_counts)} query counts in {query_counts_path}.")
            if len(np.unique(query_counts)) >= 2:
                query_slope, query_intercept = np.polyfit(query_counts, times, deg=1)
                query_fit_x = np.linspace(query_counts.min(), query_counts.max(), 200)
                query_fit_y = query_slope * query_fit_x + query_intercept

        peak_alloc_mb = None
        if peak_alloc_path is not None:
            peak_alloc_mb = _load_memory_trace_mb(peak_alloc_path)
            peak_alloc_mb = _align_memory_trace(peak_alloc_mb, len(hit_counts), peak_alloc_path)

        peak_reserved_mb = None
        if peak_reserved_path is not None:
            peak_reserved_mb = _load_memory_trace_mb(peak_reserved_path)
            peak_reserved_mb = _align_memory_trace(peak_reserved_mb, len(hit_counts), peak_reserved_path)
        else:
            peak_reserved_mb = _load_summary_peak_mb(memory_dir, model_name, metric="reserved")

        reserved_fit = None
        if isinstance(peak_reserved_mb, np.ndarray):
            reserved_fit = _fit_trend_line(hit_counts, peak_reserved_mb)

        if peak_alloc_mb is not None or peak_reserved_mb is not None:
            any_memory = True

        series.append(
            {
                "label": label,
                "model_name": model_name,
                "timing_dir": timing_dir,
                "times": times,
                "hit_counts": hit_counts,
                "query_counts": query_counts,
                "slope": slope,
                "intercept": intercept,
                "fit_x": fit_x,
                "fit_y": fit_y,
                "query_slope": query_slope,
                "query_intercept": query_intercept,
                "query_fit_x": query_fit_x,
                "query_fit_y": query_fit_y,
                "peak_alloc_mb": peak_alloc_mb,
                "peak_reserved_mb": peak_reserved_mb,
                "reserved_fit": reserved_fit,
            }
        )

    fig, ax_time = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    for idx, run in enumerate(series):
        colour = colours[idx % len(colours)]
        ax_time.scatter(
            run["hit_counts"],
            run["times"],
            color=colour,
            s=22,
            alpha=0.32,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ax_time.plot(
            run["fit_x"],
            run["fit_y"],
            color=colour,
            linewidth=2.4,
            zorder=4,
        )

    model_handles = _make_model_handles(series, colours)
    ax_time.legend(
        handles=model_handles,
        ncol=min(3, len(model_handles)),
        frameon=False,
        loc="upper left",
    )

    ax_time.set_ylabel("Inference time [ms]")
    ax_time.set_title("Inference latency scaling")
    ax_time.set_xlabel("Hits per event")
    _style_axis(ax_time)

    latency_output_path = _resolve_output_path(output, series)
    fig.savefig(latency_output_path, dpi=300)
    plt.close(fig)

    memory_output_path = None
    if any_memory:
        fig, ax_mem = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        for idx, run in enumerate(series):
            colour = colours[idx % len(colours)]
            if run["peak_alloc_mb"] is not None:
                ax_mem.scatter(
                    run["hit_counts"],
                    run["peak_alloc_mb"],
                    color=colour,
                    s=28,
                    alpha=0.45,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
            if run["reserved_fit"] is not None:
                fit_x, fit_y = run["reserved_fit"]
                ax_mem.plot(
                    fit_x,
                    fit_y,
                    color=colour,
                    linewidth=2.6,
                    alpha=0.98,
                    zorder=4,
                )
            elif run["peak_reserved_mb"] is not None:
                ax_mem.hlines(
                    float(run["peak_reserved_mb"]),
                    xmin=float(run["hit_counts"].min()),
                    xmax=float(run["hit_counts"].max()),
                    colors=colour,
                    linewidth=2.6,
                    alpha=0.98,
                    zorder=4,
                )

        memory_min_values = []
        memory_max_values = []
        for run in series:
            if run["peak_alloc_mb"] is not None:
                memory_min_values.append(float(np.min(run["peak_alloc_mb"])))
                memory_max_values.append(float(np.max(run["peak_alloc_mb"])))
            if run["peak_reserved_mb"] is not None:
                if isinstance(run["peak_reserved_mb"], np.ndarray):
                    memory_min_values.append(float(np.min(run["peak_reserved_mb"])))
                    memory_max_values.append(float(np.max(run["peak_reserved_mb"])))
                else:
                    memory_min_values.append(float(run["peak_reserved_mb"]))
                    memory_max_values.append(float(run["peak_reserved_mb"]))

        if memory_max_values:
            memory_min = min(memory_min_values) if memory_min_values else 0.0
            memory_max = max(memory_max_values)
            memory_span = max(memory_max - memory_min, 1.0)
            lower_margin = min(20.0, 0.3 * memory_span)
            upper_margin = max(40.0, 0.3 * memory_span)
            ax_mem.set_ylim(bottom=max(0.0, memory_min - lower_margin), top=memory_max + upper_margin)

        model_legend = ax_mem.legend(
            handles=model_handles,
            frameon=False,
            loc="upper left",
            ncol=min(3, len(model_handles)),
        )
        ax_mem.add_artist(model_legend)
        ax_mem.legend(
            handles=_make_memory_encoding_handles(),
            frameon=False,
            loc="upper right",
        )
        ax_mem.set_xlabel("Hits per event")
        ax_mem.set_ylabel("Peak GPU memory [MB]")
        ax_mem.set_title("GPU memory footprint")
        _style_axis(ax_mem)
        memory_output_path = _make_related_output_path(latency_output_path, "memory")
        fig.savefig(memory_output_path, dpi=300)
        plt.close(fig)

    query_series = [run for run in series if run["query_counts"] is not None]
    query_output_path = None
    if query_series:
        query_output_path = _make_related_output_path(latency_output_path, "queries")
        fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        for idx, run in enumerate(query_series):
            colour = colours[idx % len(colours)]
            ax.scatter(
                run["query_counts"],
                run["times"],
                color=colour,
                s=22,
                alpha=0.32,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )
            if run["query_fit_x"] is not None and run["query_fit_y"] is not None:
                ax.plot(
                    run["query_fit_x"],
                    run["query_fit_y"],
                    color=colour,
                    linewidth=2.4,
                    zorder=4,
                )

        ax.set_xlabel("Dynamic queries per event")
        ax.set_ylabel("Inference time [ms]")
        ax.set_title("Inference latency vs dynamic query count")
        _style_axis(ax)
        ax.legend(handles=_make_model_handles(query_series, colours), frameon=False, loc="upper left")
        fig.savefig(query_output_path, dpi=300)
        plt.close(fig)

    table_path = latency_output_path.with_suffix(".csv")
    with table_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_label",
                "event_index",
                "hits",
                "query_count",
                "inference_time_ms",
                "fit_slope_ms_per_hit",
                "fit_intercept_ms",
                "query_fit_slope_ms_per_query",
                "query_fit_intercept_ms",
            ]
        )
        for run in series:
            for event_index, (hit_count, time_ms) in enumerate(zip(run["hit_counts"], run["times"], strict=True)):
                query_count = "" if run["query_counts"] is None else int(run["query_counts"][event_index])
                query_slope = "" if run["query_slope"] is None else float(run["query_slope"])
                query_intercept = "" if run["query_intercept"] is None else float(run["query_intercept"])
                writer.writerow(
                    [
                        run["label"],
                        event_index,
                        int(hit_count),
                        query_count,
                        float(time_ms),
                        float(run["slope"]),
                        float(run["intercept"]),
                        query_slope,
                        query_intercept,
                    ]
                )

    print(f"Saved latency plot to {latency_output_path}")
    if memory_output_path is not None:
        print(f"Saved memory plot to {memory_output_path}")
    if query_output_path is not None:
        print(f"Saved query timing plot to {query_output_path}")
    print(f"Saved per-event timing table to {table_path}")
    for run in series:
        print(f'{run["label"]}: time_ms = {run["slope"]:.6f} * hits + {run["intercept"]:.6f}')
        if run["query_counts"] is not None and run["query_slope"] is not None:
            print(f'{run["label"]}: time_ms = {run["query_slope"]:.6f} * queries + {run["query_intercept"]:.6f}')
    if any_memory:
        print("Saved allocated-memory scatter points and reserved-memory trend lines to a separate memory figure.")


if __name__ == "__main__":
    main()
