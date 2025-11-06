"""
Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former
"""

import scipy
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.matcher import Matcher


# @torch.compile
def batch_dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the DICE loss for the entire batch, similar to generalized IOU for masks, over all permutations of the inputs and targets."""

    inputs = inputs.sigmoid()

    # inputs has shape (B, N, C), targets has shape (B, M, C)
    # We want to compute the DICE loss for each combination of N and M for each batch

    # Using torch.einsum to handle the batched matrix multiplication
    numerator = 2 * torch.einsum("bnc,bmc->bnm", inputs, targets)

    # Compute the denominator using sum over the last dimension (C) and broadcasting
    denominator = inputs.sum(-1).unsqueeze(2) + targets.sum(-1).unsqueeze(1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss


# @torch.compile
def batch_sigmoid_ce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the cross entropy loss over all permutations of the inputs and targets."""
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("bnc,bmc->bnm", pos, targets) + torch.einsum("bnc,bmc->bnm", neg, (1 - targets))
    return loss / inputs.shape[2]


# @torch.compile
def batch_sigmoid_focal_cost(inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2) -> Tensor:
    """Compute the focal loss over all permutations of the inputs and targets for batches."""
    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)
    loss = torch.einsum("bnc,bmc->bnm", focal_pos, targets) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - targets))
    return loss / inputs.shape[2]


@torch.jit.script
def dice_loss(inputs: Tensor, labels: Tensor):
    """Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        labels: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * labels).sum(-1)
    denominator = inputs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / len(inputs)


@torch.jit.script
def mask_ce_loss(inputs: Tensor, labels: Tensor):
    """Args:
    ----
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        labels: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns
    -------
        Loss tensor.
    """
    # seems straightforward to mask this one, just with weight argument / making it nn / setting to zero
    loss = F.binary_cross_entropy_with_logits(inputs, labels, reduction="none")  # , pos_weight=Tensor([5e3], device=inputs.device))

    # find the mean loss for each mask
    loss = loss.mean(1)

    # take the average over all masks
    return loss.sum() / len(inputs)


@torch.jit.script
def sigmoid_focal_loss(inputs: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / len(inputs)


class HEPFormerLoss(nn.Module):
    """Compute the loss of MaskFormer, based on DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the preds of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box).
    """

    def __init__(
        self,
        num_classes: int,
        num_objects: int,
        loss_weights: dict,
        matcher_weights: dict | None = None,
        null_class_weight: float = 0.5,
        losses: list[str] | None = None,
        tasks: nn.ModuleList = None,
        adaptive_lap: bool = True,
    ):
        """Create the criterion.

        Parameters
        ----------
        num_classes: int
            number of object categories, omitting the special no-object category
        num_objects: int
            number of objects to detect
        loss_weights: dict
            dict containing as key the names of the losses and as values their relative weight
        matcher_weights: dict, optional
            same as loss_weights but for the matching cost
        null_class_weight: float, optional
            relative classification weight applied to the no-object category
        losses: list, optional
            list of all the losses to be applied. See get_loss for list of available losses
        tasks: list, optional
            list of additional configurable tasks to use in the loss calculation
        """

        super().__init__()
        self.num_classes = num_classes
        self.null_class_weight = null_class_weight
        assert self.num_classes > 0
        if self.num_classes == 1:
            empty_weight = torch.tensor([self.null_class_weight])
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.null_class_weight
        self.register_buffer("empty_weight", empty_weight)
        self.loss_weights = loss_weights
        if matcher_weights is None:
            matcher_weights = loss_weights
        self.losses = losses if losses is not None else ["labels", "masks"]
        self.tasks = tasks

        self.matcher = Matcher(
            default_solver="scipy",  # or "lap1015_late" if available
            adaptive_solver=True,
            parallel_solver=False,
        )
        self.matcher_weights = matcher_weights

    def loss_labels(self, preds, labels):
        """Classification loss (NLL)
        labels dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes].
        """
        # use the new indices to calculate the loss
        # process full inidices
        flav_pred_logits = preds["track_valid"]["track_logit"].flatten(0, 1)
        flavour_labels = labels["particle_valid"].flatten(0, 1)
        loss = F.binary_cross_entropy_with_logits(flav_pred_logits.squeeze(), flavour_labels.float(), pos_weight=self.empty_weight)

        return {"object_class_ce": loss}

    def loss_masks(self, preds, labels):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        labels dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # select valid masks via flavour label
        valid_idx = labels["particle_valid"]
        target_masks = labels["particle_hit_valid"][valid_idx].float()
        pred_masks = preds["track_hit_valid"]["track_hit_logit"][valid_idx]
        # pred_masks = preds["masks"][valid_idx][:, : target_masks.shape[-1]] # for padding

        # compute losses on valid masks
        losses = {}
        if self.loss_weights.get("mask_dice"):
            losses["mask_dice"] = dice_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_focal"):
            losses["mask_focal"] = sigmoid_focal_loss(pred_masks, target_masks)
        if self.loss_weights.get("mask_ce"):
            losses["mask_ce"] = mask_ce_loss(pred_masks, target_masks)
        return losses

    def get_loss(self, loss, preds, labels):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return self.weight_loss(loss_map[loss](preds, labels))

    def weight_loss(self, losses: dict):
        """Apply the loss weights to the loss dict."""
        for k in list(losses.keys()):
            if k in self.loss_weights:
                losses[k] *= self.loss_weights[k]
        return losses

    def compute_cost_matrix(self, preds, targets):
        dev = preds["track_hit_valid"]["track_hit_logit"].device
        valid_obj_mask = targets["particle_valid"]

        # Detached copies for matching only
        mask_pred_detached = preds["track_hit_valid"]["track_hit_logit"].detach()
        mask_tgt_detached = targets["particle_hit_valid"].detach().to(mask_pred_detached.dtype)

        obj_class_pred_detached = preds["track_valid"]["track_prob"].detach().squeeze(-1)
        obj_class_tgt = valid_obj_mask.long()

        # Compute class-level cost
        gathered = (
            obj_class_pred_detached.unsqueeze(2) * obj_class_tgt.unsqueeze(1)
            + (1 - obj_class_pred_detached.unsqueeze(2)) * (1 - obj_class_tgt.unsqueeze(1))
        )
        output = gathered.squeeze(-1)

        valid_mask = valid_obj_mask.unsqueeze(1).expand(-1, obj_class_pred_detached.size(1), -1)
        output = output * valid_mask
        obj_class_cost = -output
        cost_total = self.matcher_weights["object_class_ce"] * obj_class_cost

        # Add mask-based costs — also detached!
        if self.matcher_weights.get("mask_dice"):
            cost_mask_dice = batch_dice_cost(mask_pred_detached, mask_tgt_detached)
            cost_total += cost_mask_dice * self.matcher_weights["mask_dice"]

        if self.matcher_weights.get("mask_ce"):
            cost_mask_ce = batch_sigmoid_ce_cost(mask_pred_detached, mask_tgt_detached)
            cost_total += cost_mask_ce * self.matcher_weights["mask_ce"]

        if self.matcher_weights.get("mask_focal"):
            cost_mask_focal = batch_sigmoid_focal_cost(mask_pred_detached, mask_tgt_detached)
            cost_total += cost_mask_focal * self.matcher_weights["mask_focal"]

        return cost_total, valid_obj_mask

    # def compute_cost_matrix(self, preds, targets):
    #     dev = preds["track_hit_valid"]["track_hit_logit"].device
    #     valid_obj_mask = targets["particle_valid"].detach()
    #     mask_pred = preds["track_hit_valid"]["track_hit_logit"].detach()
    #     mask_tgt = targets["particle_hit_valid"].detach().to(mask_pred.dtype)

    #     # compute per-pair cost components
    #     cost_total = None
    #     obj_class_pred = preds["track_valid"]["track_prob"].sigmoid().detach()
    #     obj_class_tgt = valid_obj_mask.long()  # [B, N_tgt]

    #     # Expand predictions and targets for pairwise comparison
    #     obj_class_pred_exp = obj_class_pred.unsqueeze(2).expand(-1, -1, obj_class_tgt.size(1), -1)  # [B, N_pred, N_tgt, 1]
    #     obj_class_tgt_exp = obj_class_tgt.unsqueeze(1).expand(-1, obj_class_pred.size(1), -1).unsqueeze(-1)  # [B, N_pred, N_tgt, 1]

    #     # Compute probability of the correct class directly
    #     gathered = obj_class_pred_exp * obj_class_tgt_exp + (1 - obj_class_pred_exp) * (1 - obj_class_tgt_exp)  # [B, N_pred, N_tgt, 1]
    #     output = gathered.squeeze(-1)  # [B, N_pred, N_tgt]

    #     # Mask invalid targets
    #     valid_mask = valid_obj_mask.unsqueeze(1).expand(-1, obj_class_pred.size(1), -1)  # [B, N_pred, N_tgt]
    #     output = output * valid_mask

    #     # Compute class cost (negative probability)
    #     obj_class_cost = -output  # [B, N_pred, N_tgt]

    #     # Weighted
    #     cost_total = self.matcher_weights["object_class_ce"] * obj_class_cost

    #     if self.matcher_weights.get("mask_dice"):
    #         cost_mask_dice = batch_dice_cost(mask_pred, mask_tgt)
    #         cost_total = (
    #             cost_mask_dice * self.matcher_weights["mask_dice"]
    #             if cost_total is None
    #             else cost_total + cost_mask_dice * self.matcher_weights["mask_dice"]
    #         )
    #     if self.matcher_weights.get("mask_ce"):
    #         cost_mask_ce = batch_sigmoid_ce_cost(mask_pred, mask_tgt)
    #         cost_total = (
    #             cost_mask_ce * self.matcher_weights["mask_ce"] if cost_total is None else cost_total + cost_mask_ce * self.matcher_weights["mask_ce"]
    #         )
    #     if self.matcher_weights.get("mask_focal"):
    #         cost_mask_focal = batch_sigmoid_focal_cost(mask_pred, mask_tgt)
    #         cost_total = (
    #             cost_mask_focal * self.matcher_weights["mask_focal"]
    #             if cost_total is None
    #             else cost_total + cost_mask_focal * self.matcher_weights["mask_focal"]
    #         )

    #     return cost_total, valid_obj_mask

    def forward(self, preds, labels):
        """Calculate the maskformer loss via optimal assignment."""

        losses = {}

        # TODO add back in
        # # loop over intermediate outputs and compute losses
        for layer_name, layer_outputs in preds.items():
            layer_costs = None

            # Skip tasks that do not contribute intermediate losses
            if layer_name == "final":
                continue

            cost_matrix, valid_mask = self.compute_cost_matrix(layer_outputs, labels)
            pred_idxs = self.matcher(cost_matrix, valid_mask)
            batch_idx = torch.arange(pred_idxs.shape[0]).unsqueeze(-1)
            aux_idx = (batch_idx, pred_idxs)
            
            layer_outputs["track_valid"]["track_prob"] = preds[layer_name]["track_valid"]["track_prob"][batch_idx, pred_idxs]
            layer_outputs["track_hit_valid"]["track_hit_logit"] = preds[layer_name]["track_hit_valid"]["track_hit_logit"][batch_idx, pred_idxs]
            layer_outputs["track_valid"]["track_logit"] = preds[layer_name]["track_valid"]["track_logit"][batch_idx, pred_idxs]

            for loss in self.losses:
                l_dict = self.get_loss(loss, layer_outputs, labels)
                l_dict = {k + f"_layer{layer_name}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Sanity check dice/focal components
        mask_pred, mask_tgt = preds["final"]["track_hit_valid"]["track_hit_logit"], labels["particle_hit_valid"]

        # get the optimal assignment of the predictions to the labels
        # idx = self.matcher(preds, labels)
        # Compute cost and perform matching
        cost_matrix, valid_mask = self.compute_cost_matrix(preds["final"], labels)
        pred_idxs = self.matcher(cost_matrix, valid_mask)
        batch_idx = torch.arange(labels["particle_valid"].shape[0]).unsqueeze(1)

        # warning: don't put this into a function or comprehension

        preds["final"]["track_valid"]["track_prob"] = preds["final"]["track_valid"]["track_prob"][batch_idx, pred_idxs]
        preds["final"]["track_hit_valid"]["track_hit_logit"] = preds["final"]["track_hit_valid"]["track_hit_logit"][batch_idx, pred_idxs]
        preds["final"]["track_valid"]["track_logit"] = preds["final"]["track_valid"]["track_logit"][batch_idx, pred_idxs]

        # compute the requested losses
        for loss in self.losses:
            losses.update(self.get_loss(loss, preds["final"], labels))

        return preds, labels, losses


