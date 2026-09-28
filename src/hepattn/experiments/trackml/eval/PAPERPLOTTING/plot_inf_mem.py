import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


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


def _get_series_family(label: str) -> str:
    parts = label.split("-", maxsplit=1)
    return parts[1] if len(parts) == 2 else label


def _get_series_architecture(label: str) -> str:
    upper_label = label.upper()
    if "LCA" in upper_label:
        return "LCA"
    if "MA" in upper_label:
        return "MA"
    parts = label.split("-", maxsplit=1)
    return parts[0].upper()


def _build_family_colour_map(series: list[dict], fallback_colours: list[str]) -> dict[str, str]:
    families: list[str] = []
    for run in series:
        family = _get_series_family(run["label"])
        if family not in families:
            families.append(family)
    return {family: fallback_colours[idx % len(fallback_colours)] for idx, family in enumerate(families)}


def _get_series_style(label: str, family_colours: dict[str, str]) -> dict[str, object]:
    architecture = _get_series_architecture(label)
    family = _get_series_family(label)
    return {
        "color": family_colours[family],
        "linestyle": "-" if architecture == "MA" else "--" if architecture == "LCA" else "-.",
        "marker": "o" if architecture == "MA" else "D" if architecture == "LCA" else "s",
    }


def _make_model_handles(series: list[dict], family_colours: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=_get_series_style(run["label"], family_colours)["color"],
            linestyle=_get_series_style(run["label"], family_colours)["linestyle"],
            marker=_get_series_style(run["label"], family_colours)["marker"],
            markersize=6,
            markerfacecolor=_get_series_style(run["label"], family_colours)["color"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=2.8,
            label=run["label"],
        )
        for run in series
    ]


def _add_model_legend(axis, series: list[dict], family_colours: dict[str, str]):
    return axis.legend(
        handles=_make_model_handles(series, family_colours),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        borderaxespad=0.0,
        ncol=1,
    )


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
            label="Allocated",
        ),
        Line2D([0], [0], color="#444444", linewidth=2.4, label="Reserved"),
    ]


def _resolve_output_path(output: Path | None, series: list[dict]) -> Path:
    if output is None:
        if len(series) == 1:
            return series[0]["timing_dir"] / f'{series[0]["model_name"]}_time_vs_hits.pdf'
        return Path.cwd() / "trackml_multi_model_time_vs_hits.pdf"

    if output.suffix:
        return output

    output.mkdir(parents=True, exist_ok=True)
    stem = "trackml_inference_time_vs_hits"
    if len(series) == 1:
        stem = f'{series[0]["model_name"]}_time_vs_hits'
    elif len(series) > 1:
        stem = "trackml_multi_model_time_vs_hits"
    return output / f"{stem}.pdf"


def _make_related_output_path(base_path: Path, target: str) -> Path:
    stem = base_path.stem
    if "time_vs_hits" in stem:
        stem = stem.replace("time_vs_hits", f"{target}_vs_hits")
    else:
        stem = f"{stem}_{target}_vs_hits"
    return base_path.with_name(f"{stem}{base_path.suffix}")


def _fit_trend_line(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    if len(xs) < 2 or len(np.unique(xs)) < 2:
        return None
    slope, intercept = np.polyfit(xs, ys, deg=1)
    fit_x = np.linspace(xs.min(), xs.max(), 200)
    fit_y = slope * fit_x + intercept
    return float(slope), float(intercept), fit_x, fit_y


def _default_hits_axis_segments(series: list[dict]) -> list[tuple[float, float]]:
    all_hits = np.concatenate([run["hit_counts"] for run in series]).astype(np.float64)
    hit_min = float(np.min(all_hits))
    hit_max = float(np.max(all_hits))
    hit_span = max(hit_max - hit_min, 1.0)
    pad = max(0.04 * hit_span, 0.02 * hit_max)
    return [(max(0.0, hit_min - pad), hit_max + pad)]


def _create_hits_axes(x_segments: list[tuple[float, float]], figsize: tuple[float, float]) -> tuple[plt.Figure, list]:
    if len(x_segments) == 1:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.set_xlim(*x_segments[0])
        return fig, [ax]

    spans = [max(right - left, 1.0) for left, right in x_segments]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figsize[0] + 1.2, figsize[1]),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": spans},
    )
    for ax, (xmin, xmax) in zip(axes, x_segments, strict=True):
        ax.set_xlim(xmin, xmax)

    axes[0].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(labelleft=False, left=False)

    d = 0.012
    kwargs = dict(transform=axes[0].transAxes, color="k", clip_on=False, linewidth=0.9)
    axes[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axes[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs = dict(transform=axes[1].transAxes, color="k", clip_on=False, linewidth=0.9)
    axes[1].plot((-d, +d), (-d, +d), **kwargs)
    axes[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
    return fig, list(axes)


def _style_hits_ticks(axis) -> None:
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))


def _set_memory_axis_limits(axes: list, min_values: list[float], max_values: list[float]) -> None:
    if not max_values:
        return

    memory_min = min(min_values) if min_values else 0.0
    memory_max = max(max_values)
    memory_span = max(memory_max - memory_min, 1.0)
    lower_margin = min(20.0, 0.3 * memory_span)
    upper_margin = max(80.0, 0.5 * memory_span)
    for ax in axes:
        ax.set_ylim(bottom=max(0.0, memory_min - lower_margin), top=memory_max + upper_margin)


def main() -> None:

    colours = _apply_paper_style()
    # timing_dirs = [Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/"),  Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/"),  Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/")]
    # memory_dirs = [Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/")]
    # model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual"), Path("TRK-v8-eta4-pt600-epochs10"), Path("TRK-v8-eta4-lca-is-first-threshold-0p3")]
    # labels = ["MA-900", "LSCA-900", "MA-600", "LSCA-600"]
    # output = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/inference-time-lca-600/")
    x_axis_segments = None

    timing_dirs = [Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/times/")]
    memory_dirs = [Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/"), Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/logs/memory/")]
    model_names = [Path("TRK-v8-3l-900-loss-weights-25-epochs30"), Path("TRK-v8-eta4-lca-eta4-900-th0p25-w32-qual"),]
    labels = ["MA-900", "LSCA-900"]
    output = Path("/share/rcif2/pduckett/hepattn-dq/src/hepattn/experiments/trackml/eval/inference-time-lca-900/")


    # Example manual split axis:
    # x_axis_segments = [(0, 30_000), (320_000, 620_000)]

    print("model names: ", model_names)

    series = []
    any_memory = False
    for model_name_override, label_override, timing_dir, memory_dir in zip(model_names, labels, timing_dirs, memory_dirs, strict=True):
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

        time_fit = _fit_trend_line(hit_counts, times)
        if time_fit is None:
            raise ValueError(f"Could not fit timing trend for {label}: need at least 2 distinct hit counts.")
        slope, intercept, fit_x, fit_y = time_fit

        query_counts = None
        query_slope = None
        query_intercept = None
        query_fit_x = None
        query_fit_y = None
        if query_counts_path.exists():
            query_counts = _load_query_counts(query_counts_path)
            if len(query_counts) != len(times):
                raise ValueError(f"Length mismatch: {len(times)} times in {times_path} vs {len(query_counts)} query counts in {query_counts_path}.")
            query_fit = _fit_trend_line(query_counts, times)
            if query_fit is not None:
                query_slope, query_intercept, query_fit_x, query_fit_y = query_fit

        peak_alloc_mb = None
        if peak_alloc_path is not None:
            peak_alloc_mb = _load_memory_trace_mb(peak_alloc_path)
            peak_alloc_mb = _align_memory_trace(peak_alloc_mb, len(hit_counts), peak_alloc_path)
        alloc_query_fit = None
        alloc_query_slope = None
        alloc_query_intercept = None
        if peak_alloc_mb is not None and query_counts is not None:
            alloc_query_fit = _fit_trend_line(query_counts, peak_alloc_mb)
            if alloc_query_fit is not None:
                alloc_query_slope, alloc_query_intercept, _, _ = alloc_query_fit

        peak_reserved_mb = None
        if peak_reserved_path is not None:
            peak_reserved_mb = _load_memory_trace_mb(peak_reserved_path)
            peak_reserved_mb = _align_memory_trace(peak_reserved_mb, len(hit_counts), peak_reserved_path)
        else:
            peak_reserved_mb = _load_summary_peak_mb(memory_dir, model_name, metric="reserved")

        reserved_slope = None
        reserved_intercept = None
        reserved_fit = None
        if isinstance(peak_reserved_mb, np.ndarray):
            reserved_fit = _fit_trend_line(hit_counts, peak_reserved_mb)
            if reserved_fit is not None:
                reserved_slope, reserved_intercept, _, _ = reserved_fit
        elif peak_reserved_mb is not None:
            reserved_slope = 0.0
            reserved_intercept = float(peak_reserved_mb)

        reserved_query_slope = None
        reserved_query_intercept = None
        reserved_query_fit = None
        if isinstance(peak_reserved_mb, np.ndarray) and query_counts is not None:
            reserved_query_fit = _fit_trend_line(query_counts, peak_reserved_mb)
            if reserved_query_fit is not None:
                reserved_query_slope, reserved_query_intercept, _, _ = reserved_query_fit
        elif peak_reserved_mb is not None and query_counts is not None:
            reserved_query_slope = 0.0
            reserved_query_intercept = float(peak_reserved_mb)

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
                "alloc_query_slope": alloc_query_slope,
                "alloc_query_intercept": alloc_query_intercept,
                "alloc_query_fit": alloc_query_fit,
                "peak_reserved_mb": peak_reserved_mb,
                "reserved_slope": reserved_slope,
                "reserved_intercept": reserved_intercept,
                "reserved_fit": reserved_fit,
                "reserved_query_slope": reserved_query_slope,
                "reserved_query_intercept": reserved_query_intercept,
                "reserved_query_fit": reserved_query_fit,
            }
        )

    family_colours = _build_family_colour_map(series, colours)
    x_segments = _default_hits_axis_segments(series) if x_axis_segments is None else x_axis_segments

    fig, time_axes = _create_hits_axes(x_segments, figsize=(7.0, 4.4))
    for ax in time_axes:
        for run in series:
            style = _get_series_style(run["label"], family_colours)
            mask = (run["hit_counts"] >= ax.get_xlim()[0]) & (run["hit_counts"] <= ax.get_xlim()[1])
            if np.any(mask):
                ax.scatter(
                    run["hit_counts"][mask],
                    run["times"][mask],
                    color=style["color"],
                    marker=style["marker"],
                    s=22,
                    alpha=0.32,
                    edgecolors="white",
                    linewidths=0.45,
                    zorder=3,
                )
            ax.plot(
                run["fit_x"],
                run["fit_y"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                zorder=4,
            )
            _style_axis(ax)
            _style_hits_ticks(ax)

    model_handles = _make_model_handles(series, family_colours)
    _add_model_legend(time_axes[0], series, family_colours)

    time_axes[0].set_ylabel("Inference time [ms]")
    if len(time_axes) == 1:
        time_axes[0].set_xlabel("Hits per event")
    else:
        fig.supxlabel("Hits per event")

    latency_output_path = _resolve_output_path(output, series)
    fig.savefig(latency_output_path, dpi=300)
    plt.close(fig)

    memory_output_path = None
    if any_memory:
        fig, memory_axes = _create_hits_axes(x_segments, figsize=(7.0, 4.4))
        for ax in memory_axes:
            for run in series:
                style = _get_series_style(run["label"], family_colours)
                mask = (run["hit_counts"] >= ax.get_xlim()[0]) & (run["hit_counts"] <= ax.get_xlim()[1])
                if run["peak_alloc_mb"] is not None and np.any(mask):
                    ax.scatter(
                        run["hit_counts"][mask],
                        run["peak_alloc_mb"][mask],
                        color=style["color"],
                        marker=style["marker"],
                        s=28,
                        alpha=0.45,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=3,
                    )
                if run["reserved_fit"] is not None:
                    _, _, fit_x, fit_y = run["reserved_fit"]
                    ax.plot(
                        fit_x,
                        fit_y,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2.6,
                        alpha=0.98,
                        zorder=4,
                    )
                elif run["peak_reserved_mb"] is not None:
                    segment_xmin = max(float(run["hit_counts"].min()), ax.get_xlim()[0])
                    segment_xmax = min(float(run["hit_counts"].max()), ax.get_xlim()[1])
                    if segment_xmin >= segment_xmax:
                        continue
                    ax.hlines(
                        float(run["peak_reserved_mb"]),
                        xmin=segment_xmin,
                        xmax=segment_xmax,
                        colors=style["color"],
                        linestyles=style["linestyle"],
                        linewidth=2.6,
                        alpha=0.98,
                        zorder=4,
                    )
                _style_axis(ax)
                _style_hits_ticks(ax)

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

        _set_memory_axis_limits(memory_axes, memory_min_values, memory_max_values)

        model_legend = _add_model_legend(memory_axes[0], series, family_colours)
        if memory_axes[0] is memory_axes[-1]:
            memory_axes[0].add_artist(model_legend)

        memory_axes[-1].legend(
            handles=_make_memory_encoding_handles(),
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.80, 0.98),
            borderaxespad=0.0,
            title="Memory",
        )
        memory_axes[0].set_ylabel("Peak GPU memory [MB]")
        if len(memory_axes) == 1:
            memory_axes[0].set_xlabel("Hits per event")
        else:
            fig.supxlabel("Hits per event")
        memory_output_path = _make_related_output_path(latency_output_path, "memory")
        fig.savefig(memory_output_path, dpi=300)
        plt.close(fig)

    query_series = [run for run in series if run["query_counts"] is not None]
    query_output_path = None
    query_memory_output_path = None
    if query_series:
        query_output_path = _make_related_output_path(latency_output_path, "queries")
        fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        for run in query_series:
            style = _get_series_style(run["label"], family_colours)
            ax.scatter(
                run["query_counts"],
                run["times"],
                color=style["color"],
                marker=style["marker"],
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
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.4,
                    zorder=4,
                )

        ax.set_xlabel("Dynamic queries per event")
        ax.set_ylabel("Inference time [ms]")
        _style_axis(ax)
        _style_hits_ticks(ax)
        _add_model_legend(ax, query_series, family_colours)
        fig.savefig(query_output_path, dpi=300)
        plt.close(fig)

        query_memory_series = [
            run for run in query_series if run["peak_alloc_mb"] is not None or run["peak_reserved_mb"] is not None
        ]
        if query_memory_series:
            query_memory_output_path = _make_related_output_path(latency_output_path, "memory_vs_queries")
            fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
            for run in query_memory_series:
                style = _get_series_style(run["label"], family_colours)
                if run["peak_alloc_mb"] is not None:
                    ax.scatter(
                        run["query_counts"],
                        run["peak_alloc_mb"],
                        color=style["color"],
                        marker=style["marker"],
                        s=28,
                        alpha=0.45,
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=3,
                    )
                    if run["alloc_query_fit"] is not None:
                        _, _, fit_x, fit_y = run["alloc_query_fit"]
                        ax.plot(
                            fit_x,
                            fit_y,
                            color=style["color"],
                            linestyle=":",
                            linewidth=1.8,
                            alpha=0.9,
                            zorder=4,
                        )
                if run["reserved_query_fit"] is not None:
                    _, _, fit_x, fit_y = run["reserved_query_fit"]
                    ax.plot(
                        fit_x,
                        fit_y,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2.6,
                        alpha=0.98,
                        zorder=5,
                    )
                elif run["peak_reserved_mb"] is not None:
                    ax.hlines(
                        float(run["peak_reserved_mb"]),
                        xmin=float(np.min(run["query_counts"])),
                        xmax=float(np.max(run["query_counts"])),
                        colors=style["color"],
                        linestyles=style["linestyle"],
                        linewidth=2.6,
                        alpha=0.98,
                        zorder=5,
                    )

            query_memory_min_values = []
            query_memory_max_values = []
            for run in query_memory_series:
                if run["peak_alloc_mb"] is not None:
                    query_memory_min_values.append(float(np.min(run["peak_alloc_mb"])))
                    query_memory_max_values.append(float(np.max(run["peak_alloc_mb"])))
                if run["peak_reserved_mb"] is not None:
                    if isinstance(run["peak_reserved_mb"], np.ndarray):
                        query_memory_min_values.append(float(np.min(run["peak_reserved_mb"])))
                        query_memory_max_values.append(float(np.max(run["peak_reserved_mb"])))
                    else:
                        query_memory_min_values.append(float(run["peak_reserved_mb"]))
                        query_memory_max_values.append(float(run["peak_reserved_mb"]))

            _set_memory_axis_limits([ax], query_memory_min_values, query_memory_max_values)

            ax.set_xlabel("Dynamic queries per event")
            ax.set_ylabel("Peak GPU memory [MB]")
            _style_axis(ax)
            _style_hits_ticks(ax)
            model_legend = _add_model_legend(ax, query_memory_series, family_colours)
            ax.add_artist(model_legend)
            ax.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color="#444444",
                        linestyle=":",
                        linewidth=1.8,
                        label="Allocated fit",
                    ),
                    *_make_memory_encoding_handles(),
                ],
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(0.68, 0.98),
                borderaxespad=0.0,
                title="Memory",
            )
            fig.savefig(query_memory_output_path, dpi=300)
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
    if query_memory_output_path is not None:
        print(f"Saved query memory plot to {query_memory_output_path}")
    print(f"Saved per-event timing table to {table_path}")
    for run in series:
        print(f'{run["label"]}: timing slope = {run["slope"]:.6f} ms/hit; fit: time_ms = {run["slope"]:.6f} * hits + {run["intercept"]:.6f}')
        if run["query_counts"] is not None and run["query_slope"] is not None:
            print(
                f'{run["label"]}: query-timing slope = {run["query_slope"]:.6f} ms/query; '
                f'fit: time_ms = {run["query_slope"]:.6f} * queries + {run["query_intercept"]:.6f}'
            )
        if run["peak_reserved_mb"] is not None and run["reserved_slope"] is not None:
            print(
                f'{run["label"]}: reserved-memory slope = {run["reserved_slope"]:.6f} MB/hit; '
                f'fit: memory_mb = {run["reserved_slope"]:.6f} * hits + {run["reserved_intercept"]:.6f}'
            )
        if run["peak_alloc_mb"] is not None and run["alloc_query_slope"] is not None:
            print(
                f'{run["label"]}: allocated-memory/query slope = {run["alloc_query_slope"]:.6f} MB/query; '
                f'fit: alloc_memory_mb = {run["alloc_query_slope"]:.6f} * queries + {run["alloc_query_intercept"]:.6f}'
            )
        if run["peak_reserved_mb"] is not None and run["reserved_query_slope"] is not None:
            print(
                f'{run["label"]}: reserved-memory/query slope = {run["reserved_query_slope"]:.6f} MB/query; '
                f'fit: reserved_memory_mb = {run["reserved_query_slope"]:.6f} * queries + {run["reserved_query_intercept"]:.6f}'
            )
    if any_memory:
        print("Saved allocated-memory scatter points and reserved-memory trend lines to a separate memory figure.")


if __name__ == "__main__":
    main()
