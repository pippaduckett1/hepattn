import json
import os
import socket
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.utils.cuda_timer import cuda_timer


class InferenceTimer(Callback):
    def __init__(
        self,
        repeats_per_batch: int = 20,
        warmup_repeats_per_batch: int = 5,
        drop_first_n_batches: int = 0,
        warmup_full_test_set_passes: int = 0,
        summary_stat: str = "median",
        measurement_full_test_set_passes: int = 0,
        measurement_order: str = "cyclic",
        measurement_burn_in_batches: int = 0,
        measurement_seed: int = 12345,
    ):
        super().__init__()

        if repeats_per_batch < 1:
            raise ValueError("repeats_per_batch must be at least 1.")
        if warmup_repeats_per_batch < 0:
            raise ValueError("warmup_repeats_per_batch must be non-negative.")
        if drop_first_n_batches < 0:
            raise ValueError("drop_first_n_batches must be non-negative.")
        if warmup_full_test_set_passes < 0:
            raise ValueError("warmup_full_test_set_passes must be non-negative.")
        if summary_stat not in {"median", "mean"}:
            raise ValueError("summary_stat must be either 'median' or 'mean'.")
        if measurement_full_test_set_passes < 0:
            raise ValueError("measurement_full_test_set_passes must be non-negative.")
        if measurement_order not in {"sequential", "shuffle", "cyclic"}:
            raise ValueError("measurement_order must be 'sequential', 'shuffle', or 'cyclic'.")
        if measurement_burn_in_batches < 0:
            raise ValueError("measurement_burn_in_batches must be non-negative.")
        if measurement_full_test_set_passes and repeats_per_batch != 1:
            raise ValueError("Sweep timing requires repeats_per_batch=1; one forward is timed per event per sweep.")
        if measurement_full_test_set_passes and warmup_repeats_per_batch:
            raise ValueError("Sweep timing requires warmup_repeats_per_batch=0; use full-test-set warm-up passes instead.")
        if measurement_full_test_set_passes and drop_first_n_batches:
            raise ValueError("Sweep timing requires drop_first_n_batches=0; use measurement_burn_in_batches instead.")

        self.repeats_per_batch = repeats_per_batch
        self.warmup_repeats_per_batch = warmup_repeats_per_batch
        self.n_warm_start = drop_first_n_batches
        self.warmup_full_test_set_passes = warmup_full_test_set_passes
        self.summary_stat = summary_stat
        self.measurement_full_test_set_passes = measurement_full_test_set_passes
        self.measurement_order = measurement_order
        self.measurement_burn_in_batches = measurement_burn_in_batches
        self.measurement_seed = measurement_seed

        self._is_warmup_pass = False
        self._wrapped_module = None
        self._tmp_dims = None
        self._tmp_batch_size = None
        self._warned_on_batch_size = False
        self._cuda = False
        self._device = None
        self._sweep_measurement_active = False
        self._sweep_measurement_complete = False
        self._sweep_last_measurement = None
        self._sweep_times = None
        self._sweep_positions = None

        self._reset_measurements()

    def _reset_measurements(self):
        self.times = []
        self.repeat_times = []
        self.dims = []
        self.batch_sizes = []
        self.query_counts = []
        self.peak_allocated = []
        self.peak_reserved = []
        self.repeat_peak_allocated = []
        self.repeat_peak_reserved = []
        self._tmp_dims = None
        self._tmp_batch_size = None
        self._sweep_measurement_active = False
        self._sweep_measurement_complete = False
        self._sweep_last_measurement = None
        self._sweep_times = None
        self._sweep_positions = None

    def _aggregate(self, values: list[float | int]) -> float:
        array = np.asarray(values, dtype=np.float64)
        if self.summary_stat == "mean":
            return float(array.mean())
        return float(np.median(array))

    def _extract_input_tensor(self, args, kwargs):
        if args and isinstance(args[0], dict):
            return args[0]
        if isinstance(kwargs.get("inputs"), dict):
            return kwargs["inputs"]
        return None

    def _capture_forward_metadata(self, args, kwargs):
        inputs = self._extract_input_tensor(args, kwargs)
        if inputs is None:
            return

        hit_valid = inputs.get("hit_valid")
        if hit_valid is None:
            return

        self._tmp_dims = int(hit_valid.shape[-1])
        self._tmp_batch_size = int(hit_valid.shape[0])

        if self._tmp_batch_size != 1 and not self._warned_on_batch_size:
            warnings.warn(
                "InferenceTimer is timing one dataloader batch per measurement. "
                "For per-event timing, ensure the test dataloader yields one event per batch.",
                UserWarning,
            )
            self._warned_on_batch_size = True

    def _measure_forward_once(self, measure: bool, args, kwargs):
        if self._cuda:
            torch.cuda.synchronize(self._device)

        base_alloc = 0
        base_rsvd = 0
        if self._cuda and measure:
            torch.cuda.reset_peak_memory_stats(self._device)
            base_alloc = int(torch.cuda.memory_allocated(self._device))
            base_rsvd = int(torch.cuda.memory_reserved(self._device))

        if self._cuda:
            timer_bucket = [] if not measure else None
            timed_values = [] if measure else timer_bucket
            with cuda_timer(timed_values, self._device):
                out = self.old_forward(*args, **kwargs)
            elapsed_ms = timed_values[0] if measure else None
        else:
            start_time = time.perf_counter()
            out = self.old_forward(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0 if measure else None

        peak_alloc = None
        peak_rsvd = None
        if self._cuda and measure:
            torch.cuda.synchronize(self._device)
            peak_alloc = max(int(torch.cuda.max_memory_allocated(self._device)) - base_alloc, 0)
            peak_rsvd = max(int(torch.cuda.max_memory_reserved(self._device)) - base_rsvd, 0)

        return out, elapsed_ms, peak_alloc, peak_rsvd

    def _run_repeated_forward(self, args, kwargs):
        if self._is_warmup_pass:
            return self.old_forward(*args, **kwargs)

        batch_repeat_times = []
        batch_repeat_allocated = []
        batch_repeat_reserved = []
        out = None

        total_repeats = self.warmup_repeats_per_batch + self.repeats_per_batch
        for repeat_idx in range(total_repeats):
            measure = repeat_idx >= self.warmup_repeats_per_batch
            out, elapsed_ms, peak_alloc, peak_rsvd = self._measure_forward_once(measure=measure, args=args, kwargs=kwargs)

            if not measure:
                continue

            batch_repeat_times.append(float(elapsed_ms))
            if peak_alloc is not None and peak_rsvd is not None:
                batch_repeat_allocated.append(int(peak_alloc))
                batch_repeat_reserved.append(int(peak_rsvd))

        if not batch_repeat_times:
            raise ValueError("No inference timings were recorded for the current batch.")

        self.repeat_times.append(batch_repeat_times)
        self.times.append(self._aggregate(batch_repeat_times))

        if batch_repeat_allocated:
            self.repeat_peak_allocated.append(batch_repeat_allocated)
            self.repeat_peak_reserved.append(batch_repeat_reserved)
            self.peak_allocated.append(self._aggregate(batch_repeat_allocated))
            self.peak_reserved.append(self._aggregate(batch_repeat_reserved))

        return out

    def _extract_query_count(self, pl_module, test_step_outputs):
        model_outputs = test_step_outputs[0] if isinstance(test_step_outputs, (tuple, list)) else test_step_outputs

        if isinstance(model_outputs, dict):
            encoder_outputs = model_outputs.get("encoder", {})
            query_mask = encoder_outputs.get("query_mask")
            if query_mask is not None:
                return int(query_mask.to(dtype=torch.int64).sum().item())

        model = getattr(pl_module, "model", pl_module)
        decoder = getattr(model, "decoder", None)
        static_num_queries = getattr(decoder, "_num_queries", None)
        if static_num_queries is None:
            return None
        return int(static_num_queries)

    def _model_inputs_from_batch(self, batch):
        if isinstance(batch, dict):
            return batch
        if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], dict):
            return batch[0]
        raise TypeError("InferenceTimer requires a dict input batch or a batch whose first item is an input dict.")

    def _iter_test_dataloaders(self, trainer):
        dataloaders = getattr(trainer, "test_dataloaders", None)
        if dataloaders is None:
            return []
        if isinstance(dataloaders, (list, tuple)):
            return list(dataloaders)
        return [dataloaders]

    def _run_full_test_set_warmup(self, trainer, pl_module) -> None:
        if self.warmup_full_test_set_passes == 0:
            return

        dataloaders = self._iter_test_dataloaders(trainer)
        if not dataloaders:
            raise ValueError("InferenceTimer requested full-test-set warmup, but no test dataloaders were found.")

        self._is_warmup_pass = True
        try:
            with torch.inference_mode():
                for pass_idx in range(self.warmup_full_test_set_passes):
                    print(
                        f"InferenceTimer full-test-set warm-up pass "
                        f"{pass_idx + 1}/{self.warmup_full_test_set_passes}"
                    )
                    for dataloader_idx, dataloader in enumerate(dataloaders):
                        for batch in dataloader:
                            batch = trainer.strategy.batch_to_device(batch, device=self._device, dataloader_idx=dataloader_idx)
                            with trainer.precision_plugin.test_step_context():
                                self._wrapped_module(self._model_inputs_from_batch(batch))
        finally:
            self._is_warmup_pass = False
            self._reset_measurements()

    def _sweep_order(self, num_events: int, sweep_idx: int, rng: np.random.Generator) -> np.ndarray:
        event_indices = np.arange(num_events, dtype=np.int64)
        if self.measurement_order == "shuffle":
            return rng.permutation(event_indices)
        if self.measurement_order == "cyclic":
            return np.roll(event_indices, -(sweep_idx % num_events))
        return event_indices

    def _run_sweep_forward(self, args, kwargs):
        out, elapsed_ms, peak_alloc, peak_rsvd = self._measure_forward_once(measure=True, args=args, kwargs=kwargs)
        if self._sweep_last_measurement is not None:
            raise RuntimeError("Sweep timing expects exactly one model.forward call per test_step.")
        self._sweep_last_measurement = (float(elapsed_ms), peak_alloc, peak_rsvd)
        return out

    def _benchmark_dataloader(self, trainer, order: np.ndarray):
        datamodule = getattr(trainer, "datamodule", None)
        benchmark_dataloader = getattr(datamodule, "benchmark_test_dataloader", None)
        if benchmark_dataloader is None:
            raise ValueError(
                "Sweep timing requires the datamodule to implement benchmark_test_dataloader(indices)."
            )
        return benchmark_dataloader(order.tolist())

    def _run_full_test_set_measurements(self, trainer, pl_module) -> None:
        if self.measurement_full_test_set_passes == 0:
            return

        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "test_dataset", None)
        if dataset is None:
            raise ValueError("Sweep timing requires an initialized map-style test_dataset.")
        num_events = len(dataset)
        if num_events == 0:
            raise ValueError("Sweep timing requires at least one test event.")
        if self.measurement_burn_in_batches >= num_events:
            raise ValueError("measurement_burn_in_batches must be smaller than the number of test events.")

        num_sweeps = self.measurement_full_test_set_passes
        sweep_times = np.full((num_events, num_sweeps), np.nan, dtype=np.float64)
        sweep_positions = np.full((num_events, num_sweeps), -1, dtype=np.int64)
        dims = np.full(num_events, -1, dtype=np.int64)
        batch_sizes = np.full(num_events, -1, dtype=np.int64)
        query_counts = np.full(num_events, -1, dtype=np.int64)
        rng = np.random.default_rng(self.measurement_seed)

        with torch.inference_mode():
            for sweep_idx in range(num_sweeps):
                order = self._sweep_order(num_events=num_events, sweep_idx=sweep_idx, rng=rng)
                dataloader = self._benchmark_dataloader(trainer=trainer, order=order)
                print(
                    f"InferenceTimer timed sweep {sweep_idx + 1}/{num_sweeps} "
                    f"({self.measurement_order}, burn-in {self.measurement_burn_in_batches} events)"
                )
                for position, (event_idx, batch) in enumerate(zip(order, dataloader, strict=True)):
                    self._tmp_dims = None
                    self._tmp_batch_size = None
                    self._sweep_last_measurement = None
                    self._sweep_measurement_active = position >= self.measurement_burn_in_batches
                    batch = trainer.strategy.batch_to_device(batch, device=self._device, dataloader_idx=0)
                    try:
                        with trainer.precision_plugin.test_step_context():
                            outputs = self._wrapped_module(self._model_inputs_from_batch(batch))
                    finally:
                        self._sweep_measurement_active = False

                    event_idx = int(event_idx)
                    if self._tmp_dims is not None:
                        dims[event_idx] = self._tmp_dims
                    if self._tmp_batch_size is not None:
                        batch_sizes[event_idx] = self._tmp_batch_size
                    query_count = self._extract_query_count(pl_module, outputs)
                    if query_count is not None:
                        query_counts[event_idx] = query_count

                    if position < self.measurement_burn_in_batches:
                        continue
                    if self._sweep_last_measurement is None:
                        raise RuntimeError("Sweep timing did not observe model.forward for a measured test_step.")
                    elapsed_ms, _, _ = self._sweep_last_measurement
                    sweep_times[event_idx, sweep_idx] = elapsed_ms
                    sweep_positions[event_idx, sweep_idx] = position

        if np.any(dims < 0) or np.any(batch_sizes < 0) or np.any(query_counts < 0):
            raise RuntimeError("Sweep timing could not collect event metadata for every test event.")

        aggregate = np.nanmean(sweep_times, axis=1) if self.summary_stat == "mean" else np.nanmedian(sweep_times, axis=1)
        self.times = aggregate.tolist()
        self.repeat_times = []
        self.dims = dims.tolist()
        self.batch_sizes = batch_sizes.tolist()
        self.query_counts = query_counts.tolist()
        self._sweep_times = sweep_times
        self._sweep_positions = sweep_positions
        self._sweep_measurement_complete = True

    def _compiled_model_enabled(self, pl_module) -> bool:
        model = getattr(pl_module, "model", pl_module)
        if hasattr(model, "_orig_mod"):
            return True
        return any(hasattr(module, "_orig_mod") for module in model.modules())

    def _write_metadata(self, trainer, pl_module):
        metadata = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "run_name": pl_module.name,
            "device": str(self._device),
            "device_name": str(self._device) if self._cuda else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "trainer_precision": str(getattr(trainer, "precision", "unknown")),
            "repeats_per_batch": self.repeats_per_batch,
            "warmup_repeats_per_batch": self.warmup_repeats_per_batch,
            "drop_first_n_batches": self.n_warm_start,
            "warmup_full_test_set_passes": self.warmup_full_test_set_passes,
            "summary_stat": self.summary_stat,
            "measurement_mode": "full_test_set_sweeps" if self.measurement_full_test_set_passes else "per_batch_repeats",
            "measurement_full_test_set_passes": self.measurement_full_test_set_passes,
            "measurement_order": self.measurement_order,
            "measurement_burn_in_batches": self.measurement_burn_in_batches,
            "measurement_seed": self.measurement_seed,
            "num_recorded_batches": len(self.times),
            "compiled_model_enabled": self._compiled_model_enabled(pl_module),
            "timed_region": "wrapped model.forward only; dataloading and Lightning bookkeeping excluded",
        }

        with (self.times_path / f"{pl_module.name}_timing_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def _infer_device(self, trainer, pl_module) -> torch.device | None:
        root_device = getattr(trainer.strategy, "root_device", None)
        if root_device is not None:
            return root_device

        for tensor in list(pl_module.parameters()) + list(pl_module.buffers()):
            return tensor.device

        return None

    def on_test_start(self, trainer, pl_module):
        assert trainer.global_rank == 0, "InferenceTimer should only be used with a single process."
        self._reset_measurements()

        model = pl_module
        if hasattr(model, "model"):
            model = model.model
        self._wrapped_module = model
        self.old_forward = model.forward

        self._device = self._infer_device(trainer=trainer, pl_module=pl_module)
        self._cuda = self._device is not None and self._device.type == "cuda"
        if self._cuda:
            self._device = torch.device("cuda", self._device.index if self._device.index is not None else 0)

        def new_forward(*args, **kwargs):
            self._capture_forward_metadata(args, kwargs)
            if self._sweep_measurement_active:
                return self._run_sweep_forward(args, kwargs)
            if self._sweep_measurement_complete:
                return self.old_forward(*args, **kwargs)
            return self._run_repeated_forward(args, kwargs)

        model.forward = new_forward

        matmul_precision = torch.get_float32_matmul_precision()
        if matmul_precision in {"high", "highest"}:
            warnings.warn(
                f"""The current float32 matmul precision is set to {matmul_precision},
            which may impact inference times. Consider if `low` or `medium` matmul
            precision can be used instead.""",
                UserWarning,
            )

        self._run_full_test_set_warmup(trainer=trainer, pl_module=pl_module)
        self._run_full_test_set_measurements(trainer=trainer, pl_module=pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._sweep_measurement_complete:
            self._tmp_dims = None
            self._tmp_batch_size = None
            return
        if self._is_warmup_pass:
            self._tmp_dims = None
            self._tmp_batch_size = None
            return

        if self._tmp_dims is not None:
            self.dims.append(self._tmp_dims)
            self._tmp_dims = None

        if self._tmp_batch_size is not None:
            self.batch_sizes.append(self._tmp_batch_size)
            self._tmp_batch_size = None

        query_count = self._extract_query_count(pl_module, outputs)
        if query_count is not None:
            self.query_counts.append(query_count)

    def on_test_end(self, trainer, pl_module):
        if self._wrapped_module is not None:
            self._wrapped_module.forward = self.old_forward

        if not self.times:
            raise ValueError("No times recorded.")

        if not self._sweep_measurement_complete:
            self.times = self.times[self.n_warm_start :]
            self.repeat_times = self.repeat_times[self.n_warm_start :]
            self.dims = self.dims[self.n_warm_start :]
            self.batch_sizes = self.batch_sizes[self.n_warm_start :]
            self.query_counts = self.query_counts[self.n_warm_start :]

            if self._cuda:
                self.peak_allocated = self.peak_allocated[self.n_warm_start :]
                self.peak_reserved = self.peak_reserved[self.n_warm_start :]
                self.repeat_peak_allocated = self.repeat_peak_allocated[self.n_warm_start :]
                self.repeat_peak_reserved = self.repeat_peak_reserved[self.n_warm_start :]

        if not self.times:
            print("Not enough steps to obtain timing information")
            return

        self.times = torch.tensor(self.times, dtype=torch.float64)
        repeat_times_array = (
            self._sweep_times if self._sweep_measurement_complete else np.asarray(self.repeat_times, dtype=np.float64)
        )
        self.mean_time = float(self.times.mean().item())
        self.median_time = float(self.times.median().item())
        self.std_time = float(self.times.std(unbiased=False).item()) if len(self.times) > 1 else 0.0
        self.mean_repeat_std = float(np.nanmean(np.nanstd(repeat_times_array, axis=1))) if repeat_times_array.shape[1] > 1 else 0.0

        if self._cuda and self.peak_allocated:
            alloc = torch.tensor(self.peak_allocated, dtype=torch.float64)
            rsvd = torch.tensor(self.peak_reserved, dtype=torch.float64)
            self.mean_peak_alloc_mb = alloc.mean().item() / (1024**2)
            self.max_peak_alloc_mb = alloc.max().item() / (1024**2)
            self.mean_peak_rsvd_mb = rsvd.mean().item() / (1024**2)
            self.max_peak_rsvd_mb = rsvd.max().item() / (1024**2)

        log_dir = trainer.log_dir or trainer.default_root_dir
        self.times_path = Path(log_dir) / "times"
        self.times_path.mkdir(parents=True, exist_ok=True)

        np.save(self.times_path / f"{pl_module.name}_times.npy", self.times.cpu().numpy())
        np.save(self.times_path / f"{pl_module.name}_repeat_times_ms.npy", repeat_times_array)
        np.save(self.times_path / f"{pl_module.name}_dims.npy", np.asarray(self.dims, dtype=np.int64))
        np.save(self.times_path / f"{pl_module.name}_batch_sizes.npy", np.asarray(self.batch_sizes, dtype=np.int64))
        np.save(self.times_path / f"{pl_module.name}_query_counts.npy", np.asarray(self.query_counts, dtype=np.int64))
        if self._sweep_measurement_complete:
            np.save(self.times_path / f"{pl_module.name}_sweep_positions.npy", self._sweep_positions)

        if self._cuda and self.repeat_peak_allocated:
            np.save(
                self.times_path / f"{pl_module.name}_peak_allocated_bytes.npy",
                np.asarray(self.peak_allocated, dtype=np.float64),
            )
            np.save(
                self.times_path / f"{pl_module.name}_peak_reserved_bytes.npy",
                np.asarray(self.peak_reserved, dtype=np.float64),
            )
            np.save(
                self.times_path / f"{pl_module.name}_repeat_peak_allocated_bytes.npy",
                np.asarray(self.repeat_peak_allocated, dtype=np.int64),
            )
            np.save(
                self.times_path / f"{pl_module.name}_repeat_peak_reserved_bytes.npy",
                np.asarray(self.repeat_peak_reserved, dtype=np.int64),
            )

        self._write_metadata(trainer=trainer, pl_module=pl_module)

    def teardown(self, trainer, pl_module, stage):
        if len(self.times):
            print("-" * 80)
            measurement_description = (
                f"{self.summary_stat} across retained sweep measurements"
                if self._sweep_measurement_complete
                else f"{self.summary_stat} of {self.repeats_per_batch} repeats after {self.warmup_repeats_per_batch} warmups"
            )
            print(
                f"Inference time per batch ({measurement_description}): "
                f"median {self.median_time:.2f} ms, mean {self.mean_time:.2f} ms, std across batches {self.std_time:.2f} ms"
            )
            if self.repeats_per_batch > 1 or self._sweep_measurement_complete:
                print(f"Mean within-batch repeat std: {self.mean_repeat_std:.2f} ms")

            if getattr(self, "_cuda", False) and len(getattr(self, "peak_allocated", [])):
                print(
                    f"Peak GPU mem per batch (allocated, {self.summary_stat} across repeats): "
                    f"mean {self.mean_peak_alloc_mb:.1f} MB, max {self.max_peak_alloc_mb:.1f} MB"
                )
                print(
                    f"Peak GPU mem per batch (reserved, {self.summary_stat} across repeats):  "
                    f"mean {self.mean_peak_rsvd_mb:.1f} MB, max {self.max_peak_rsvd_mb:.1f} MB"
                )

            print(f"Saved timing info to {self.times_path}")
            print("-" * 80)
