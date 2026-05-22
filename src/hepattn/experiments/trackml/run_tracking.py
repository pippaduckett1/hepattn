import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import Tensor, nn

from hepattn.experiments.trackml.data import TrackMLDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLTracker(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def _metric_targets(self, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        metric_target_keys = (
            "particle_valid",
            "particle_hit_valid",
            "hit_on_valid_particle",
            "hit_is_first",
            "hit_is_last",
            "key_on_valid_particle",
            "key_is_first",
            "key_is_last",
        )
        if not any(f"{key}_eval" in targets for key in metric_target_keys):
            return targets

        metric_targets = dict(targets)
        for key in metric_target_keys:
            eval_key = f"{key}_eval"
            if eval_key in targets:
                metric_targets[key] = targets[eval_key]
        return metric_targets

    def log_metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor], stage: str) -> None:
        super().log_metrics(preds, self._metric_targets(targets), stage)

    def log_custom_metrics(self, preds, targets, stage):
        query_mask = targets.get("query_mask")
        if query_mask is not None:
            query_mask = query_mask.bool()

        # log intermediate layer mask predictions
        for layer_name, layer_preds in preds.items():
            # Skip layers that don't have track_hit_valid task (e.g., encoder layer)
            if "track_hit_valid" not in layer_preds:
                continue
            mask = layer_preds["track_hit_valid"]["track_hit_valid"]
            if mask is not None:
                num_valid = mask.sum(-1).float()
                frac_valid = num_valid / mask.shape[-1]

                if query_mask is None:
                    self.log(f"{stage}/{layer_name}_avg_num_valid_hits", torch.mean(num_valid), sync_dist=True)
                    self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", torch.mean(frac_valid), sync_dist=True)
                else:
                    valid_num_valid = num_valid[query_mask]
                    valid_frac_valid = frac_valid[query_mask]
                    if valid_num_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_num_valid_hits", valid_num_valid.mean(), sync_dist=True)
                    if valid_frac_valid.numel() > 0:
                        self.log(f"{stage}/{layer_name}_avg_frac_valid_hits", valid_frac_valid.mean(), sync_dist=True)

        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]

        if query_mask is not None:
            pred_valid = pred_valid & query_mask

        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)

        # Number of predicted hits on the track
        hit_p = pred_hit_masks.sum(-1)

        # True number of hits on the track
        hit_t = true_hit_masks.sum(-1)

        if "track_iou" in preds and "query_iou" in preds["track_iou"]:
            pred_iou = preds["track_iou"]["query_iou"].float()
            true_iou = hit_tp.float() / (hit_p + hit_t - hit_tp).clamp_min(1).float()
            quality_mask = pred_valid | true_valid

            if query_mask is not None:
                quality_mask = quality_mask & query_mask

            if quality_mask.any().item():
                pred_iou_valid = pred_iou[quality_mask]
                true_iou_valid = true_iou[quality_mask]
                mae = (pred_iou_valid - true_iou_valid).abs().mean()
                rmse = torch.sqrt(torch.mean((pred_iou_valid - true_iou_valid) ** 2))

                pred_centered = pred_iou_valid - pred_iou_valid.mean()
                true_centered = true_iou_valid - true_iou_valid.mean()
                corr_denom = torch.sqrt(pred_centered.square().sum() * true_centered.square().sum()).clamp_min(1e-6)
                corr = (pred_centered * true_centered).sum() / corr_denom

                self.log(f"{stage}/track_iou_mae", mae, sync_dist=True)
                self.log(f"{stage}/track_iou_rmse", rmse, sync_dist=True)
                self.log(f"{stage}/track_iou_corr", corr, sync_dist=True)

        # Calculate the efficiency and purity at differnt matching working points
        for wp in [0.5, 0.75, 1.0]:
            both_valid = true_valid & pred_valid

            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        pred_num = pred_valid.sum(-1)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        true_num = true_valid.sum(-1)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

        num_hits_total = float(pred_hit_masks.shape[-1])
        num_hits_valid = float(true_hit_masks.sum())
        num_hits_noise = num_hits_total - num_hits_valid
        self.log(f"{stage}/num_hits", num_hits_total, sync_dist=True)
        self.log(f"{stage}/num_hits_valid", num_hits_valid, sync_dist=True)
        self.log(f"{stage}/num_hits_noise", num_hits_noise, sync_dist=True)

    def _augment_test_targets_with_event_counts(
        self,
        outputs: dict[str, dict[str, Tensor]],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Attach per-event query and truth-particle counts for eval-file analysis."""
        augmented_targets = dict(targets)

        query_mask = outputs.get("encoder", {}).get("query_mask")
        if query_mask is not None:
            num_initialized_queries = query_mask.to(dtype=torch.int64).sum(dim=-1)
        else:
            decoder = getattr(self.model, "decoder", None)
            static_num_queries = getattr(decoder, "_num_queries", None)
            if static_num_queries is None:
                batch_size = targets["particle_valid"].shape[0]
                num_initialized_queries = torch.zeros(batch_size, dtype=torch.int64, device=targets["particle_valid"].device)
            else:
                batch_size = targets["particle_valid"].shape[0]
                num_initialized_queries = torch.full(
                    (batch_size,),
                    int(static_num_queries),
                    dtype=torch.int64,
                    device=targets["particle_valid"].device,
                )

        num_reconstructable_particles = targets["particle_valid"].to(dtype=torch.int64).sum(dim=-1)

        augmented_targets["num_initialized_queries"] = num_initialized_queries
        augmented_targets["num_reconstructable_particles"] = num_reconstructable_particles
        particle_valid_train = targets.get("particle_valid_train")
        if particle_valid_train is not None:
            augmented_targets["num_reconstructable_particles_train"] = particle_valid_train.to(dtype=torch.int64).sum(dim=-1)
            augmented_targets["num_reconstructable_particles_eval"] = num_reconstructable_particles
        return augmented_targets

    def test_step(
        self, batch: tuple[dict[str, Tensor], dict[str, Tensor]]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """TrackML override: include post-loss targets for prediction writing.

        The post-loss targets are sorter-aligned when a sorter is configured, which keeps
        saved targets and saved predictions on the same hit ordering in eval files.
        """
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs, targets, losses = self.model.loss(outputs, targets)
        preds = self.model.predict(outputs)
        targets = self._metric_targets(targets)
        targets = self._augment_test_targets_with_event_counts(outputs=outputs, targets=targets)
        return outputs, preds, losses, targets


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=TrackMLDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
