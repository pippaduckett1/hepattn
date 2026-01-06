import math
from typing import Any

import torch
from torch import Tensor, nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask, ObjectHitMaskTask
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormer(nn.Module):
    def __init__(
        self,
        input_nets: nn.ModuleList,
        encoder: nn.Module,
        decoder: MaskFormerDecoder,
        tasks: nn.ModuleList,
        dim: int,
        target_object: str = "particle",
        pooling: nn.Module | None = None,
        matcher: nn.Module | None = None,
        input_sort_field: str | None = None,
        sorter: nn.Module | None = None,
        unified_decoding: bool = False,
        phi_alignment: dict | None = None,
    ):
        """Initializes the MaskFormer model, which is a modular transformer-style architecture designed
        for multi-task object reconstruction with attention-based decoding and optional encoder blocks.

        Args:
            input_nets: A list of input modules, each responsible for embedding a specific constituent type.
            encoder: An optional encoder module that processes merged constituent embeddings with optional sorting.
            decoder: The decoder module that handles multi-layer decoding and task integration.
            tasks: A list of task modules, each responsible for producing and processing predictions from decoder outputs.
            dim: The dimensionality of the query and key embeddings.
            target_object: The target object name which is used to mark valid/invalid objects during matching.
            pooling: An optional pooling module used to aggregate features from the input constituents.
            matcher: A module used to match predictions to targets (e.g., using the Hungarian algorithm) for loss computation.
            input_sort_field: An optional key used to sort the input constituents (e.g., for windowed attention).
            sorter: An optional sorter module used to reorder input constituents before processing.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after encoding.
        """
        super().__init__()

        self.input_nets = input_nets
        self.encoder = encoder
        self.decoder = decoder
        self.decoder.tasks = tasks
        self.pooling = pooling
        self.tasks = tasks
        self.target_object = target_object
        self.matcher = matcher
        self.unified_decoding = unified_decoding
        self.decoder.unified_decoding = unified_decoding
        self._current_epoch: int | None = None
        self._need_query_phi = False
        self.phi_alignment_cfg = self._init_phi_alignment(phi_alignment)

        assert not (input_sort_field and sorter), "Cannot specify both input_sort_field and sorter."
        self.input_sort_field = input_sort_field
        self.sorter = sorter
        if self.sorter is not None:
            self.sorter.input_names = self.input_names

        assert "key" not in self.input_names, "'key' input name is reserved."
        assert "query" not in self.input_names, "'query' input name is reserved."
        assert not any("_" in name for name in self.input_names), "Input names cannot contain underscores."

    @property
    def input_names(self) -> list[str]:
        return [input_net.input_name for input_net in self.input_nets]

    def set_training_progress(self, epoch: int | None = None, global_step: int | None = None) -> None:
        """Expose training progress to submodules that need scheduling."""
        self._current_epoch = epoch
        if hasattr(self.decoder, "set_curriculum_progress"):
            self.decoder.set_curriculum_progress(epoch=epoch, global_step=global_step)

    def _init_phi_alignment(self, cfg: dict | None) -> dict:
        cfg = cfg or {}
        weight = float(cfg.get("weight", 0.0))
        enabled = bool(cfg.get("use", False)) and weight > 0
        result = {
            "enabled": enabled,
            "weight": weight,
            "layer": cfg.get("layer", "final"),
            "warmup_epochs": int(cfg.get("warmup_epochs", 0)),
            "input_name": cfg.get("input_name", self.input_names[0] if self.input_names else None),
            "phi_field": cfg.get("phi_field", "phi"),
            "eps": float(cfg.get("eps", 1e-6)),
            "mask_task": None,
            "logit_key": None,
        }
        if not enabled:
            return result

        mask_task_name = cfg.get("mask_task")
        task = self._find_mask_task(mask_task_name)
        if task is None:
            # Disable gracefully if we cannot find a usable mask task
            result["enabled"] = False
            return result

        result["mask_task"] = task
        result["mask_task_name"] = task.name
        result["logit_key"] = task.output_object_hit + "_logit"
        # Ensure query phi is available for loss computation
        self.decoder.log_query_phi = True
        self._need_query_phi = True
        return result

    def _find_mask_task(self, mask_task_name: str | None) -> ObjectHitMaskTask | None:
        if mask_task_name is not None:
            for task in self.tasks:
                if task.name == mask_task_name and isinstance(task, ObjectHitMaskTask):
                    return task
            return None

        for task in self.tasks:
            if isinstance(task, ObjectHitMaskTask):
                return task
        return None

    def forward(self, inputs: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        batch_size = inputs[self.input_names[0] + "_valid"].shape[0]
        x = {"inputs": inputs}

        # Embed the input constituents
        for input_net in self.input_nets:
            input_name = input_net.input_name
            x[input_name + "_embed"] = input_net(inputs)
            x[input_name + "_valid"] = inputs[input_name + "_valid"]

            # These slices can be used to pick out specific
            # objects after we have merged them all together
            # Only needed when not doing unified decoding
            if not self.unified_decoding:
                device = inputs[input_name + "_valid"].device
                mask = torch.cat([torch.full((inputs[i + "_valid"].shape[-1],), i == input_name, device=device) for i in self.input_names], dim=-1)
                x[f"key_is_{input_name}"] = mask.unsqueeze(0).expand(batch_size, -1)

        # Merge the input constituents and the padding mask into a single set
        x["key_embed"] = torch.concatenate([x[input_name + "_embed"] for input_name in self.input_names], dim=-2)
        x["key_valid"] = torch.concatenate([x[input_name + "_valid"] for input_name in self.input_names], dim=-1)

        # If all key_valid are true, then we can just set it to None, however,
        # if we are using flash-varlen, we have to always provide a kv_mask argument
        if batch_size == 1 and x["key_valid"].all() and self.encoder.attn_type != "flash-varlen":
            x["key_valid"] = None

        # LEGACY. TODO: remove
        if self.input_sort_field and not self.sorter:
            x[f"key_{self.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.input_sort_field] for input_name in self.input_names], dim=-1
            )

        # Dedicated sorting step before encoder
        if self.sorter is not None:
            x[f"key_{self.sorter.input_sort_field}"] = torch.concatenate(
                [inputs[input_name + "_" + self.sorter.input_sort_field] for input_name in self.input_names], dim=-1
            )
            for input_name in self.input_names:
                field = f"{input_name}_{self.sorter.input_sort_field}"
                x[field] = inputs[field]
            x = self.sorter.sort_inputs(x)

        # Pass merged input constituents through the encoder
        x_sort_value = x.get(f"key_{self.input_sort_field}") if self.sorter is None else None
        x["key_embed"] = self.encoder(x["key_embed"], x_sort_value=x_sort_value, kv_mask=x.get("key_valid"))

        # Unmerge the updated features back into the separate input types only if not doing unified decoding
        if not self.unified_decoding:
            x = unmerge_inputs(x, self.input_names)

        # Pass through decoder layers
        x, outputs = self.decoder(x, self.input_names)

        # Do any pooling if desired
        if self.pooling is not None:
            x_pooled = self.pooling(x[f"{self.pooling.input_name}_embed"], x[f"{self.pooling.input_name}_valid"])
            x[f"{self.pooling.output_name}_embed"] = x_pooled

        # Get the final outputs
        outputs["final"] = {}
        for task in self.tasks:
            outputs["final"][task.name] = task(x)

            # Need this for incidence-based regression task
            if isinstance(task, IncidenceRegressionTask):
                x["incidence"] = outputs["final"][task.name][task.incidence_key].detach()
            if isinstance(task, ObjectClassificationTask):
                x["class_probs"] = outputs["final"][task.name][task.probs_key].detach()

        # store info about the input sort field for each input type
        if self.sorter is not None:
            sort = self.sorter.input_sort_field
            sort_dict = {f"{name}_{sort}": inputs[f"{name}_{sort}"] for name in self.input_names}
            outputs["final"][sort] = sort_dict

        if self._need_query_phi:
            outputs["final"]["query_phi"] = x.get("query_phi")

        return outputs

    def predict(self, outputs: dict) -> dict:
        """Takes the raw model outputs and produces a set of actual inferences / predictions.
        For example will take output probabilies and apply threshold cuts to prduce boolean predictions.

        Args:
            outputs: The outputs produced by the forward pass of the model.

        Returns:
            preds: A dictionary containing the predicted values for each task.
        """
        preds: dict[str, dict[str, Any]] = {}

        # Compute predictions for each task in each block
        for layer_name, layer_outputs in outputs.items():
            preds[layer_name] = {}

            for task in self.tasks:
                if task.name not in layer_outputs:
                    continue
                preds[layer_name][task.name] = task.predict(layer_outputs[task.name])

        return preds

    def loss(self, outputs: dict, targets: dict, inputs: dict | None = None, epoch: int | None = None) -> tuple[dict, dict]:
        """Computes the loss between the forward pass of the model and the data / targets.
        It first computes the cost / loss between each of the predicted and true tracks in each ROI
        and then uses the Hungarian algorihtm to perform an optimal bipartite matching. The model
        predictions are then permuted to match this optimal matching, after which the final loss
        between the model and target is computed.

        Args:
            outputs: The outputs produced by the forward pass of the model.
            targets: The data containing the targets.

        Returns:
            losses: A dictionary containing the computed losses for each task.
        """
        # Will hold the costs between all pairs of objects - cost axes are (batch, pred, true)
        costs = {}
        if self.sorter is not None:
            targets = self.sorter.sort_targets(targets, outputs["final"][self.sorter.input_sort_field])

        batch_idxs = torch.arange(targets[f"{self.target_object}_valid"].shape[0]).unsqueeze(1)
        for layer_name, layer_outputs in outputs.items():
            layer_costs = None

            # Get the cost contribution from each of the tasks
            for task in self.tasks:
                # Skip tasks that do not contribute intermediate losses
                if task.name not in layer_outputs:
                    continue

                # Compute costs
                task_costs = task.cost(layer_outputs[task.name], targets)

                # Add the cost on to our running cost total, otherwise initialise a running cost matrix
                for cost in task_costs.values():
                    if layer_costs is None:
                        layer_costs = cost
                    else:
                        layer_costs += cost

            # Added to allow completely turning off inter layer loss
            # Possibly redundant as completely switching them off performs worse
            if layer_costs is not None:
                layer_costs = layer_costs.detach()

            costs[layer_name] = layer_costs

        # Check if we're using dynamic queries (num_queries = num_truth_particles)
        # In this case, matching is trivial - just use identity mapping
        use_dynamic_queries = getattr(self.decoder, "dynamic_queries", False)

        # With dynamic queries, slice targets to only valid particles (no padding)
        # Only slice target object tensors, not constituent/hit tensors
        if use_dynamic_queries:
            num_valid = targets[f"{self.target_object}_valid"].sum().int().item()
            target_prefix = f"{self.target_object}_"
            sliced_targets = {}
            for k, v in targets.items():
                if k.startswith(target_prefix) and v.dim() >= 2 and v.shape[1] > num_valid:
                    # Slice object-level targets (particle_valid, particle_hit_valid, particle_pt, etc.)
                    sliced_targets[k] = v[:, :num_valid]
                else:
                    sliced_targets[k] = v
            targets = sliced_targets

        # Permute the outputs for each output in each layer
        for layer_name, cost in costs.items():
            if cost is None:
                continue

            if use_dynamic_queries:
                # With dynamic queries, num_queries = num_truth_particles
                # Use identity matching - predictions already correspond 1:1 with truth
                num_queries = cost.shape[1]
                pred_idxs = torch.arange(num_queries, device=cost.device).unsqueeze(0).expand(cost.shape[0], -1)
            else:
                # Get the indicies that can permute the predictions to yield their optimal matching
                pred_idxs = self.matcher(cost, targets[f"{self.target_object}_valid"])

            if "query_phi" in outputs[layer_name]:
                outputs[layer_name]["query_phi"] = outputs[layer_name]["query_phi"][batch_idxs, pred_idxs]

            for task in self.tasks:
                # Tasks without a object dimension do not need permutation (constituent-level or sample-level)
                if not task.permute_loss:
                    continue

                # The task didn't produce an output for this layer, so skip it
                if task.name not in outputs[layer_name]:
                    continue

                for output_name in task.outputs:
                    outputs[layer_name][task.name][output_name] = outputs[layer_name][task.name][output_name][batch_idxs, pred_idxs]

        # Compute the losses for each task in each block
        losses: dict[str, dict[str, Tensor]] = {}
        for layer_name in outputs:
            losses[layer_name] = {}
            is_final_layer = layer_name == "final"
            for task in self.tasks:
                if task.name not in outputs[layer_name]:
                    continue

                task_losses = task.loss(outputs[layer_name][task.name], targets)

                # Remove IoU loss from intermediate layers (only keep it for final layer)
                if not is_final_layer:
                    task_losses.pop("iou_mse", None)

                losses[layer_name][task.name] = task_losses

        phi_loss = self._phi_alignment_loss(outputs, targets, inputs=inputs, epoch=epoch)
        if phi_loss is not None:
            layer = self.phi_alignment_cfg["layer"]
            if layer not in losses:
                losses[layer] = {}
            losses[layer].setdefault("phi_alignment", {})
            losses[layer]["phi_alignment"]["phi_align"] = phi_loss

        return losses, targets

    def _phi_alignment_loss(self, outputs: dict, targets: dict, inputs: dict | None, epoch: int | None) -> Tensor | None:
        cfg = self.phi_alignment_cfg
        if not cfg["enabled"]:
            return None
        if inputs is None:
            return None
        if epoch is None:
            epoch = self._current_epoch
        if epoch is not None and epoch < cfg["warmup_epochs"]:
            return None

        layer = cfg["layer"]
        task = cfg["mask_task"]
        logit_key = cfg["logit_key"]
        if task is None or logit_key is None:
            return None
        if layer not in outputs or task.name not in outputs[layer]:
            return None

        layer_outputs = outputs[layer]
        logits = layer_outputs[task.name].get(logit_key)
        query_phi = layer_outputs.get("query_phi")
        if logits is None or query_phi is None:
            return None

        input_name = cfg["input_name"]
        if input_name is None:
            return None
        phi_key = f"{input_name}_{cfg['phi_field']}"
        hit_phi = inputs.get(phi_key)
        if hit_phi is None:
            return None

        hit_valid = inputs.get(f"{input_name}_valid")
        query_valid = targets.get(f"{self.target_object}_valid")

        probs = logits.sigmoid()
        if hit_valid is not None:
            probs = probs * hit_valid.unsqueeze(1).to(probs.dtype)

        phi = hit_phi.to(probs.dtype)
        sin_mean = (probs * phi.sin().unsqueeze(1)).sum(dim=-1)
        cos_mean = (probs * phi.cos().unsqueeze(1)).sum(dim=-1)
        weight_sum = probs.sum(dim=-1)
        valid_hits = weight_sum > cfg["eps"]
        weight_sum = weight_sum.clamp_min(cfg["eps"])

        mean_phi = torch.atan2(sin_mean / weight_sum, cos_mean / weight_sum)
        phi_diff = query_phi - mean_phi
        phi_diff = (phi_diff + math.pi) % (2 * math.pi) - math.pi

        if query_valid is not None:
            valid_hits = valid_hits & query_valid

        if not valid_hits.any():
            return None

        phi_loss = (phi_diff[valid_hits] ** 2).mean()
        return cfg["weight"] * phi_loss
