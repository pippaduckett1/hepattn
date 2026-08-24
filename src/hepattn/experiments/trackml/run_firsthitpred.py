import numpy as np
import torch
from lightning.pytorch.cli import ArgsType
from pathlib import Path
from torch import Tensor, nn

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLFirstHitPred(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
        topk_query_limits: list[int] | None = None,
        topk_policy: str = "global_topk",
        topk_eta_bins: int = 10,
        topk_r_bins: int = 8,
        topk_eta_range: tuple[float, float] = (-5.0, 5.0),
        topk_r_range: tuple[float, float] = (0.0, 1.5),
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

        if len(self.model.tasks) != 1:
            raise ValueError("TrackMLFirstHitPred expects exactly one task in model.tasks.")

        task = self.model.tasks[0]
        if not hasattr(task, "classes") or len(task.classes) != 1:
            raise ValueError("TrackMLFirstHitPred expects a single-class ClassificationTask (e.g. classes: ['is_first']).")

        self._task_name = task.name
        self._pred_key = f"{task.output_object}_{task.classes[0]}"
        self._prob_key = f"{task.output_object}_{task.classes[0]}_prob"
        self._target_key = f"{task.target_object}_{task.classes[0]}"
        self._valid_key = f"{task.target_object}_valid"
        self._eta_key = f"{task.input_object}_eta"
        self._r_key = f"{task.input_object}_r"
        self.topk_query_limits = sorted({int(k) for k in (topk_query_limits or []) if int(k) > 0})
        self.topk_policy = topk_policy
        self.topk_eta_bins = topk_eta_bins
        self.topk_r_bins = topk_r_bins
        self.topk_eta_range = topk_eta_range
        self.topk_r_range = topk_r_range
        if self.topk_policy not in {"global_topk", "eta_r_topk"}:
            raise ValueError(f"Unsupported topk_policy: {self.topk_policy}")
        if self.topk_eta_bins <= 0 or self.topk_r_bins <= 0:
            raise ValueError("topk_eta_bins and topk_r_bins must be positive")

        self._first_hit_eta_edges = np.linspace(-5.0, 5.0, 81)
        self._first_hit_r_edges = np.linspace(0.0, 1.5, 81)
        self._first_hit_stats: dict[str, dict[str, torch.Tensor]] = {}

    def _reset_first_hit_stats(self, stage: str) -> None:
        eta_bins = len(self._first_hit_eta_edges) - 1
        r_bins = len(self._first_hit_r_edges) - 1
        self._first_hit_stats[stage] = {
            "eta_tp": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_fn": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_fp": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_true": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_pred": torch.zeros(eta_bins, dtype=torch.float64),
            "eta_r_tp": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
            "eta_r_fn": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
            "eta_r_fp": torch.zeros((eta_bins, r_bins), dtype=torch.float64),
        }
        for k in self.topk_query_limits:
            self._first_hit_stats[stage][f"topk_{k}_policy_tp"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_policy_fp"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_policy_fn"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_policy_selected"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_global_tp"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_global_fp"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_global_fn"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_global_selected"] = torch.tensor(0.0, dtype=torch.float64)
            self._first_hit_stats[stage][f"topk_{k}_true"] = torch.tensor(0.0, dtype=torch.float64)

    def _gather_hist(self, hist: torch.Tensor) -> torch.Tensor:
        gathered = self.all_gather(hist.to(self.device))
        if gathered.dim() == hist.dim():
            return gathered.detach().cpu()
        return gathered.sum(dim=0).detach().cpu()

    def _accumulate_first_hit_hists(
        self,
        stage: str,
        eta_vals: np.ndarray,
        r_vals: np.ndarray,
        tp_mask: np.ndarray,
        fn_mask: np.ndarray,
        fp_mask: np.ndarray,
        true_mask: np.ndarray,
        pred_mask: np.ndarray,
    ) -> None:
        stats = self._first_hit_stats[stage]
        eta_edges = self._first_hit_eta_edges
        r_edges = self._first_hit_r_edges

        stats["eta_tp"] += torch.from_numpy(np.histogram(eta_vals[tp_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_fn"] += torch.from_numpy(np.histogram(eta_vals[fn_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_fp"] += torch.from_numpy(np.histogram(eta_vals[fp_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_true"] += torch.from_numpy(np.histogram(eta_vals[true_mask], bins=eta_edges)[0]).to(dtype=torch.float64)
        stats["eta_pred"] += torch.from_numpy(np.histogram(eta_vals[pred_mask], bins=eta_edges)[0]).to(dtype=torch.float64)

        stats["eta_r_tp"] += torch.from_numpy(np.histogram2d(eta_vals[tp_mask], r_vals[tp_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )
        stats["eta_r_fn"] += torch.from_numpy(np.histogram2d(eta_vals[fn_mask], r_vals[fn_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )
        stats["eta_r_fp"] += torch.from_numpy(np.histogram2d(eta_vals[fp_mask], r_vals[fp_mask], bins=[eta_edges, r_edges])[0]).to(
            dtype=torch.float64
        )

    def _extract_first_hit_masks(self, preds: dict, targets: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor] | None:
        task_preds = preds.get("final", {}).get(self._task_name)
        if task_preds is None:
            return None

        pred_first = task_preds.get(self._pred_key)
        true_first = targets.get(self._target_key)
        if pred_first is None or true_first is None:
            return None

        pred_first = pred_first.bool()
        true_first = true_first.bool()
        valid = targets.get(self._valid_key)
        if valid is None:
            valid = torch.ones_like(true_first, dtype=torch.bool)
        else:
            valid = valid.bool()

        return pred_first, true_first, valid

    def _extract_first_hit_probs(self, preds: dict, targets: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor] | None:
        task_preds = preds.get("final", {}).get(self._task_name)
        if task_preds is None:
            return None

        prob_first = task_preds.get(self._prob_key)
        true_first = targets.get(self._target_key)
        if prob_first is None or true_first is None:
            return None

        prob_first = prob_first.to(torch.float32)
        true_first = true_first.bool()
        valid = targets.get(self._valid_key)
        if valid is None:
            valid = torch.ones_like(true_first, dtype=torch.bool)
        else:
            valid = valid.bool()

        if prob_first.dim() == 1:
            prob_first = prob_first.unsqueeze(0)
            true_first = true_first.unsqueeze(0)
            valid = valid.unsqueeze(0)

        return prob_first, true_first, valid

    def _build_global_topk_mask(self, probs: Tensor, valid: Tensor, k: int) -> Tensor:
        selected = torch.zeros_like(valid, dtype=torch.bool)
        valid_idx = torch.where(valid)[0]
        if valid_idx.numel() == 0:
            return selected
        k_eff = min(int(k), int(valid_idx.numel()))
        top_local = probs[valid_idx].topk(k_eff).indices
        selected[valid_idx[top_local]] = True
        return selected

    def _build_eta_r_topk_mask(self, probs: Tensor, valid: Tensor, eta: Tensor, r: Tensor, k: int) -> Tensor:
        selected = torch.zeros_like(valid, dtype=torch.bool)
        valid_idx = torch.where(valid)[0]
        if valid_idx.numel() == 0:
            return selected
        k_eff = min(int(k), int(valid_idx.numel()))
        if k_eff == 0:
            return selected

        dtype_float = probs.dtype
        device = probs.device
        eta_edges = torch.linspace(self.topk_eta_range[0], self.topk_eta_range[1], self.topk_eta_bins + 1, device=device, dtype=dtype_float)
        r_edges = torch.linspace(self.topk_r_range[0], self.topk_r_range[1], self.topk_r_bins + 1, device=device, dtype=dtype_float)

        eta_vals = eta[valid_idx]
        r_vals = r[valid_idx]
        eta_bin = torch.bucketize(eta_vals, eta_edges, right=False) - 1
        r_bin = torch.bucketize(r_vals, r_edges, right=False) - 1
        eta_bin = eta_bin.clamp(0, self.topk_eta_bins - 1)
        r_bin = r_bin.clamp(0, self.topk_r_bins - 1)

        num_bins = self.topk_eta_bins * self.topk_r_bins
        bin_id = eta_bin * self.topk_r_bins + r_bin
        bin_counts = torch.bincount(bin_id, minlength=num_bins)
        total_candidates = int(bin_counts.sum().item())
        if total_candidates == 0:
            return self._build_global_topk_mask(probs, valid, k_eff)

        quota_float = bin_counts.to(dtype=torch.float64) * (k_eff / float(total_candidates))
        bin_quota = torch.floor(quota_float).to(dtype=torch.int64)
        bin_quota = torch.minimum(bin_quota, bin_counts.to(dtype=torch.int64))

        remaining_slots = int(k_eff - int(bin_quota.sum().item()))
        if remaining_slots > 0:
            frac = quota_float - bin_quota.to(dtype=quota_float.dtype)
            has_capacity = bin_counts.to(dtype=torch.int64) > bin_quota
            frac = torch.where(has_capacity, frac, torch.full_like(frac, -1.0))
            order = torch.argsort(frac, descending=True)
            for idx in order.tolist():
                if remaining_slots == 0:
                    break
                if frac[idx] < 0:
                    break
                if int(bin_counts[idx].item()) > int(bin_quota[idx].item()):
                    bin_quota[idx] += 1
                    remaining_slots -= 1

        for current_bin in torch.where(bin_quota > 0)[0].tolist():
            local = valid_idx[bin_id == current_bin]
            if local.numel() == 0:
                continue
            take_k = min(int(bin_quota[current_bin].item()), int(local.numel()))
            local_top = probs[local].topk(take_k).indices
            selected[local[local_top]] = True

        if int(selected.sum().item()) < k_eff:
            remaining_idx = torch.where(valid & (~selected))[0]
            if remaining_idx.numel() > 0:
                need = k_eff - int(selected.sum().item())
                top_local = probs[remaining_idx].topk(min(int(need), int(remaining_idx.numel()))).indices
                selected[remaining_idx[top_local]] = True

        return selected

    def _compute_selection_counts(self, selected: Tensor, true_first: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        selected_valid = selected & valid
        true_valid = true_first & valid
        tp = (selected_valid & true_valid).sum().float()
        fp = (selected_valid & (~true_valid)).sum().float()
        fn = ((~selected_valid) & true_valid).sum().float()
        selected_total = selected_valid.sum().float()
        true_total = true_valid.sum().float()
        return tp, fp, fn, selected_total, true_total

    def _update_topk_stats(self, inputs: dict[str, Tensor], preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        if stage not in self._first_hit_stats or len(self.topk_query_limits) == 0:
            return

        first_hit_probs = self._extract_first_hit_probs(preds, targets)
        if first_hit_probs is None:
            return
        prob_first, true_first, valid = first_hit_probs
        hit_eta = inputs.get(self._eta_key)
        hit_r = inputs.get(self._r_key)

        if hit_eta is not None and hit_eta.dim() == 1:
            hit_eta = hit_eta.unsqueeze(0)
            hit_r = hit_r.unsqueeze(0) if hit_r is not None else None

        stats = self._first_hit_stats[stage]
        for k in self.topk_query_limits:
            policy_tp = torch.tensor(0.0, device=prob_first.device)
            policy_fp = torch.tensor(0.0, device=prob_first.device)
            policy_fn = torch.tensor(0.0, device=prob_first.device)
            policy_selected_total = torch.tensor(0.0, device=prob_first.device)
            global_tp = torch.tensor(0.0, device=prob_first.device)
            global_fp = torch.tensor(0.0, device=prob_first.device)
            global_fn = torch.tensor(0.0, device=prob_first.device)
            global_selected_total = torch.tensor(0.0, device=prob_first.device)
            true_total = torch.tensor(0.0, device=prob_first.device)

            for b in range(prob_first.shape[0]):
                global_selected = self._build_global_topk_mask(prob_first[b], valid[b], k)
                g_tp, g_fp, g_fn, g_selected, g_true = self._compute_selection_counts(global_selected, true_first[b], valid[b])
                global_tp += g_tp
                global_fp += g_fp
                global_fn += g_fn
                global_selected_total += g_selected
                true_total += g_true

                if self.topk_policy == "eta_r_topk" and hit_eta is not None and hit_r is not None:
                    policy_selected = self._build_eta_r_topk_mask(prob_first[b], valid[b], hit_eta[b], hit_r[b], k)
                else:
                    policy_selected = global_selected
                p_tp, p_fp, p_fn, p_selected, _ = self._compute_selection_counts(policy_selected, true_first[b], valid[b])
                policy_tp += p_tp
                policy_fp += p_fp
                policy_fn += p_fn
                policy_selected_total += p_selected

            stats[f"topk_{k}_policy_tp"] += policy_tp.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_policy_fp"] += policy_fp.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_policy_fn"] += policy_fn.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_policy_selected"] += policy_selected_total.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_global_tp"] += global_tp.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_global_fp"] += global_fp.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_global_fn"] += global_fn.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_global_selected"] += global_selected_total.detach().to(dtype=torch.float64).cpu()
            stats[f"topk_{k}_true"] += true_total.detach().to(dtype=torch.float64).cpu()

    def _update_first_hit_stats(self, inputs: dict[str, Tensor], preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        if stage not in self._first_hit_stats:
            return

        self._update_topk_stats(inputs, preds, targets, stage)

        first_hit_masks = self._extract_first_hit_masks(preds, targets)
        if first_hit_masks is None:
            return
        pred_first, true_first, valid = first_hit_masks

        hit_eta = inputs.get(self._eta_key)
        hit_r = inputs.get(self._r_key)
        if hit_eta is None or hit_r is None:
            return

        tp = pred_first & true_first & valid
        fn = (~pred_first) & true_first & valid
        fp = pred_first & (~true_first) & valid
        true_mask = true_first & valid
        pred_mask = pred_first & valid

        eta_vals = hit_eta.detach().to(torch.float32).cpu().numpy().reshape(-1)
        r_vals = hit_r.detach().to(torch.float32).cpu().numpy().reshape(-1)

        self._accumulate_first_hit_hists(
            stage=stage,
            eta_vals=eta_vals,
            r_vals=r_vals,
            tp_mask=tp.detach().cpu().numpy().reshape(-1),
            fn_mask=fn.detach().cpu().numpy().reshape(-1),
            fp_mask=fp.detach().cpu().numpy().reshape(-1),
            true_mask=true_mask.detach().cpu().numpy().reshape(-1),
            pred_mask=pred_mask.detach().cpu().numpy().reshape(-1),
        )

    def _save_first_hit_plots(self, stage: str, stats: dict[str, torch.Tensor]) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        out_dir = Path(self.trainer.default_root_dir) / "first_hit_diagnostics" / f"{stage}_epoch{int(self.current_epoch):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        eta_edges = self._first_hit_eta_edges
        r_edges = self._first_hit_r_edges
        eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
        eta_width = eta_edges[1] - eta_edges[0]

        eta_tp = stats["eta_tp"].numpy()
        eta_fn = stats["eta_fn"].numpy()
        eta_fp = stats["eta_fp"].numpy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(eta_centers, eta_tp, where="mid", label="TP (correct first-hit predictions)")
        ax.step(eta_centers, eta_fn, where="mid", label="FN (missed true first hits)")
        ax.step(eta_centers, eta_fp, where="mid", label="FP (incorrect first-hit predictions)")
        ax.set_xlabel("eta")
        ax.set_ylabel("count")
        ax.set_title(f"{stage}: first-hit prediction outcomes vs eta")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "first_hit_outcomes_eta.png", dpi=180)
        plt.close(fig)

        eta_true = stats["eta_true"].numpy()
        eta_pred = stats["eta_pred"].numpy()
        eta_recall = np.divide(eta_tp, eta_true, out=np.zeros_like(eta_tp, dtype=float), where=eta_true > 0)
        eta_precision = np.divide(eta_tp, eta_pred, out=np.zeros_like(eta_tp, dtype=float), where=eta_pred > 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(eta_centers - 0.5 * eta_width, eta_recall, width=eta_width, label="recall", alpha=0.7)
        ax.bar(eta_centers + 0.5 * eta_width, eta_precision, width=eta_width, label="precision", alpha=0.7)
        ax.set_xlabel("eta")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{stage}: first-hit recall/precision vs eta")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "first_hit_recall_precision_eta.png", dpi=180)
        plt.close(fig)

        extent = [eta_edges[0], eta_edges[-1], r_edges[0], r_edges[-1]]
        eta_r_maps = [
            ("TP (correct)", stats["eta_r_tp"].numpy(), "Greens"),
            ("FN (missed)", stats["eta_r_fn"].numpy(), "Blues"),
            ("FP (incorrect)", stats["eta_r_fp"].numpy(), "Reds"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        for ax, (title, arr, cmap) in zip(axes, eta_r_maps):
            im = ax.imshow(arr.T, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap=cmap)
            ax.set_xlabel("eta")
            ax.set_ylabel("r [m]")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"{stage}: first-hit prediction outcomes in eta-r")
        fig.savefig(out_dir / "first_hit_outcomes_eta_r.png", dpi=180)
        plt.close(fig)

    def _finalize_first_hit_stats(self, stage: str) -> None:
        stats = self._first_hit_stats.get(stage)
        if stats is None:
            return

        gathered = {k: self._gather_hist(v) for k, v in stats.items()}
        tp_total = gathered["eta_tp"].sum()
        fn_total = gathered["eta_fn"].sum()
        fp_total = gathered["eta_fp"].sum()
        true_total = gathered["eta_true"].sum()
        pred_total = gathered["eta_pred"].sum()

        recall = tp_total / true_total if true_total > 0 else 0.0
        precision = tp_total / pred_total if pred_total > 0 else 0.0
        miss_rate = fn_total / true_total if true_total > 0 else 0.0
        false_discovery = fp_total / pred_total if pred_total > 0 else 0.0

        self.log(f"{stage}/first_hit_tp", float(tp_total), sync_dist=True)
        self.log(f"{stage}/first_hit_fn", float(fn_total), sync_dist=True)
        self.log(f"{stage}/first_hit_fp", float(fp_total), sync_dist=True)
        self.log(f"{stage}/first_hit_recall", float(recall), sync_dist=True)
        self.log(f"{stage}/first_hit_precision", float(precision), sync_dist=True)
        self.log(f"{stage}/first_hit_miss_rate", float(miss_rate), sync_dist=True)
        self.log(f"{stage}/first_hit_false_discovery", float(false_discovery), sync_dist=True)

        for k in self.topk_query_limits:
            policy_topk_tp = float(self._gather_hist(stats[f"topk_{k}_policy_tp"]).item())
            policy_topk_fp = float(self._gather_hist(stats[f"topk_{k}_policy_fp"]).item())
            policy_topk_fn = float(self._gather_hist(stats[f"topk_{k}_policy_fn"]).item())
            policy_topk_selected = float(self._gather_hist(stats[f"topk_{k}_policy_selected"]).item())
            global_topk_tp = float(self._gather_hist(stats[f"topk_{k}_global_tp"]).item())
            global_topk_fp = float(self._gather_hist(stats[f"topk_{k}_global_fp"]).item())
            global_topk_fn = float(self._gather_hist(stats[f"topk_{k}_global_fn"]).item())
            global_topk_selected = float(self._gather_hist(stats[f"topk_{k}_global_selected"]).item())
            topk_true = float(self._gather_hist(stats[f"topk_{k}_true"]).item())

            policy_topk_recall = policy_topk_tp / topk_true if topk_true > 0 else 0.0
            policy_topk_precision = policy_topk_tp / policy_topk_selected if policy_topk_selected > 0 else 0.0
            global_topk_recall = global_topk_tp / topk_true if topk_true > 0 else 0.0
            global_topk_precision = global_topk_tp / global_topk_selected if global_topk_selected > 0 else 0.0

            # Backward-compatible aliases map to the active policy.
            self.log(f"{stage}/topk_{k}_tp", policy_topk_tp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_fp", policy_topk_fp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_fn", policy_topk_fn, sync_dist=True)
            self.log(f"{stage}/topk_{k}_recall", policy_topk_recall, sync_dist=True)
            self.log(f"{stage}/topk_{k}_precision", policy_topk_precision, sync_dist=True)
            self.log(f"{stage}/topk_{k}_selected", policy_topk_selected, sync_dist=True)

            self.log(f"{stage}/topk_{k}_policy_recall", policy_topk_recall, sync_dist=True)
            self.log(f"{stage}/topk_{k}_policy_precision", policy_topk_precision, sync_dist=True)
            self.log(f"{stage}/topk_{k}_policy_tp", policy_topk_tp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_policy_fp", policy_topk_fp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_policy_fn", policy_topk_fn, sync_dist=True)

            self.log(f"{stage}/topk_{k}_global_recall", global_topk_recall, sync_dist=True)
            self.log(f"{stage}/topk_{k}_global_precision", global_topk_precision, sync_dist=True)
            self.log(f"{stage}/topk_{k}_global_tp", global_topk_tp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_global_fp", global_topk_fp, sync_dist=True)
            self.log(f"{stage}/topk_{k}_global_fn", global_topk_fn, sync_dist=True)

            self.log(f"{stage}/topk_{k}_policy_minus_global_recall", policy_topk_recall - global_topk_recall, sync_dist=True)
            self.log(f"{stage}/topk_{k}_policy_minus_global_precision", policy_topk_precision - global_topk_precision, sync_dist=True)

        if self.trainer.is_global_zero:
            self._save_first_hit_plots(stage, gathered)

    def log_custom_metrics(self, preds: dict, targets: dict[str, Tensor], stage: str) -> None:
        first_hit_masks = self._extract_first_hit_masks(preds, targets)
        if first_hit_masks is None:
            return
        pred_first, true_first, valid = first_hit_masks

        tp = (pred_first & true_first & valid).sum().float()
        fn = ((~pred_first) & true_first & valid).sum().float()
        fp = (pred_first & (~true_first) & valid).sum().float()
        tn = ((~pred_first) & (~true_first) & valid).sum().float()

        true_pos = (true_first & valid).sum().float()
        pred_pos = (pred_first & valid).sum().float()
        true_neg = ((~true_first) & valid).sum().float()
        total = valid.sum().float()

        eps = torch.tensor(1e-12, device=tp.device)
        recall = tp / torch.maximum(true_pos, eps)
        precision = tp / torch.maximum(pred_pos, eps)
        specificity = tn / torch.maximum(true_neg, eps)
        fpr = fp / torch.maximum(true_neg, eps)
        accuracy = (tp + tn) / torch.maximum(total, eps)

        self.log(f"{stage}/first_hit_batch_tp", tp, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_fn", fn, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_fp", fp, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_tn", tn, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_recall", recall, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_precision", precision, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_specificity", specificity, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_fpr", fpr, sync_dist=True, batch_size=1)
        self.log(f"{stage}/first_hit_batch_acc", accuracy, sync_dist=True, batch_size=1)


    def on_validation_epoch_start(self) -> None:
        self._reset_first_hit_stats("val")

    def on_validation_epoch_end(self) -> None:
        self._finalize_first_hit_stats("val")

    def on_test_epoch_start(self) -> None:
        self._reset_first_hit_stats("test")

    def on_test_epoch_end(self) -> None:
        self._finalize_first_hit_stats("test")

    def validation_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> dict[str, Tensor]:
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        total_loss = self.aggregate_losses(losses, stage="val")
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")
        self._update_first_hit_stats(inputs, preds, targets, stage="val")
        return {"loss": total_loss, "attn_mask_outputs": outputs}

    def test_step(self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        preds = self.model.predict(outputs)
        self._update_first_hit_stats(inputs, preds, targets, stage="test")
        return outputs, preds, losses


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLFirstHitPred,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
