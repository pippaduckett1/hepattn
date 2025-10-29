import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, loss_fns, mask_focal_loss
from hepattn.utils.masks import topk_attn
from hepattn.utils.scaling import FeatureScaler

# mypy: ignore-errors


# Mapping of loss function names to torch.nn.functional loss functions
REGRESSION_LOSS_FNS = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

# Define the literal type for regression losses based on the dictionary keys
RegressionLossType = Literal["l1", "l2", "smooth_l1"]


class Task(nn.Module, ABC):
    """Abstract base class for all tasks.

    A task represents a specific learning objective (e.g., classification, regression)
    that can be trained as part of a multi-task learning setup.
    """

    def __init__(self, has_intermediate_loss: bool, permute_loss: bool = True):
        super().__init__()
        self.has_intermediate_loss = has_intermediate_loss
        self.permute_loss = permute_loss

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the forward pass of the task."""

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def key_mask(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def query_mask(self, outputs: dict[str, Tensor], **kwargs) -> Tensor | None:
        return None


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        class_net: nn.Module = None,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_queries: bool = False,
        has_intermediate_loss: bool = True,
    ):
        """Task used for classifying whether object candidates/seeds should be taken as reconstructed/predicted objects or not.

        Args:
            name: Name of the task, used as the key to separate task outputs.
            input_object: Name of the input object.
            output_object: Name of the output object, which will denote if the predicted object slot is used or not.
            target_object: Name of the target object that we want to predict is valid or not.
            class_net: Network to project the object embedding down into a scalar.
            losses: Dict specifying which losses to use. Keys are loss function names and values are loss weights.
            costs: Dict specifying which costs to use. Keys are cost function names and values are cost weights.
            dim: Embedding dimension of the input objects.
            null_weight: Weight applied to the null class in the loss. Useful if many instances of the target class are null, and we need to reweight
                to overcome class imbalance.
            mask_queries: Whether to mask queries.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_queries = mask_queries

        # Internal
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logit"]
        self.net = class_net

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_logit = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_logit": x_logit}

    def predict(self, outputs: dict[str, Tensor], threshold: float = 0.5) -> dict[str, Tensor]:
        # Objects that have a predicted probability above the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object + "_valid"].to(torch.float32)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        sample_weight = target + self.null_weight * (1 - target)
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, sample_weight=sample_weight)
        return losses

    def query_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> Tensor | None:
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_constituent: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_attn: bool = True,
        target_field: str = "valid",
        logit_scale: float = 1.0,
        pred_threshold: float = 0.5,
        has_intermediate_loss: bool = True,
        object_net: nn.Module = None,
    ):
        """Task for predicting associations between objects and hits.

        Args:
            name: Name of the task.
            input_constituent: Name of the input constituent type (traditionally hits in tracking).
                For unified decoding, use "key" to access merged embeddings.
            input_object: Name of the input object.
            output_object: Name of the output object.
            target_object: Name of the target object.
            losses: Loss functions and their weights.
            costs: Cost functions and their weights.
            dim: Embedding dimension.
            null_weight: Weight for null class.
            mask_attn: Whether to mask attention.
            target_field: Target field name.
            logit_scale: Scale for logits.
            pred_threshold: Prediction threshold.
            has_intermediate_loss: Whether the task has intermediate loss.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss)

        self.name = name
        self.input_constituent = input_constituent
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.target_field = target_field

        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.logit_scale = logit_scale
        self.pred_threshold = pred_threshold
        self.has_intermediate_loss = mask_attn

        self.output_object_hit = output_object + "_" + input_constituent
        self.target_object_hit = target_object + "_" + input_constituent

        self.inputs = [input_object + "_embed", input_constituent + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.object_net = object_net

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the objects and hits
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = x[self.input_constituent + "_embed"]

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = self.logit_scale * torch.einsum("bqe,ble->bql", x_object, x_hit)

        # Zero out entries for any padded input constituents
        valid_key = f"{self.input_constituent}_valid"
        valid_mask = x[valid_key].unsqueeze(-2).expand_as(object_hit_logit)
        object_hit_logit[~valid_mask] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs: dict[str, Tensor], threshold: float = 0.1) -> dict[str, Tensor]:
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() < threshold
        return {self.input_constituent: attn_mask}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() > self.pred_threshold}

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object_hit + "_" + self.target_field].detach().to(output.dtype)

        hit_pad = targets[self.input_constituent + "_valid"]

        costs = {}
        # sample_weight = target + self.null_weight * (1 - target)
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target, input_pad_mask=hit_pad)
        return costs

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        hit_pad = targets[self.input_constituent + "_valid"]
        object_pad = targets[self.target_object + "_valid"]

        sample_weight = target + self.null_weight * (1 - target)
        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](
                output, target, object_valid_mask=object_pad, input_pad_mask=hit_pad, sample_weight=sample_weight
            )
        return losses
