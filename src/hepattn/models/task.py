from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, focal_loss, loss_fns
from hepattn.utils.scaling import RegressionTargetScaler
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

import sys


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: dict[str, Tensor], track_hit_valid_mask: dict[str, Tensor] | None = None, track_valid_mask: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Compute the forward pass of the task.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing the base input tensors.
        additional_inputs : dict[str, Tensor] | None, optional
            Optional dictionary containing additional input tensors.
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing the task outputs.
        """

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs, **kwargs):
        return {}

    def key_mask(self, outputs, **kwargs):
        return {}

    def query_mask(self, outputs, **kwargs):
        return None


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_queries: bool = False,
    ):
        """Task used for classifying whether object candidates / seeds should be
        taken as reconstructed / pred objects or not.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_object : str
            Name of the input object feature
        output_object : str
            Name of the output object feature which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

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
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_logit = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_logit": x_logit.squeeze(-1)}

    def predict(self, outputs, threshold=0.5):
        # Objects that have a predicted probability aove the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_logit"].detach()
        target = targets[self.target_object + "_valid"].type_as(output)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
            # Set the costs of invalid objects to be (basically) inf
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=weight)
        return losses

    def query_mask(self, outputs, threshold=0.1):
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold


class HitFilterTask(Task):
    def __init__(
        self,
        name: str,
        hit_name: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
        mask_keys: bool = False,
        loss_fn: Literal["bce", "focal"] = "bce",
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not."""
        super().__init__()

        self.name = name
        self.hit_name = hit_name
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.mask_keys = mask_keys

        # Internal
        self.input_features = [f"{hit_name}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_logit = self.net(x[f"{self.hit_name}_embed"])
        return {f"{self.hit_name}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict) -> dict:
        return {f"{self.hit_name}_{self.target_field}": outputs[f"{self.hit_name}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[f"{self.hit_name}_logit"]
        target = targets[f"{self.hit_name}_{self.target_field}"].type_as(output)

        # Ensure target has same shape as output by adding batch dimension if needed
        if output.dim() > target.dim():
            target = target.view(output.shape)  # Match exact shape
            print(f"Reshaped target to: {target.shape}")

        print(f"Final shapes - Output: {output.shape}, Target: {target.shape}")

        # Calculate the BCE loss with class weighting
        if self.loss_fn == "bce":
            weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
        elif self.loss_fn == "focal":
            loss = focal_loss(output, target)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        return {f"{self.hit_name}_{self.loss_fn}": loss}

    def key_mask(self, outputs, threshold=0.1):
        if not self.mask_keys:
            return {}

        return {self.hit_name: outputs[f"{self.hit_name}_logit"].detach().sigmoid() >= threshold}


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_attn: bool = True,
    ):
        super().__init__()

        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn

        self.output_object_hit = output_object + "_" + input_hit
        self.target_object_hit = target_object + "_" + input_hit
        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.hit_net = Dense(dim, dim)
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and objects
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = x[self.input_hit + "_embed"]

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs, threshold=0.1):
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        # TODO: See if the query masking stops this from being necessary
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"].detach()
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        print(output.shape)
        print(target.shape)


        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
            costs[cost_fn][~targets[self.target_object + "_valid"].unsqueeze(-2).expand_as(costs[cost_fn])] = 1e6
        return costs

    def loss(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        # Build a padding mask for object-hit pairs
        hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        object_pad = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(target)
        # An object-hit is valid slot if both its object and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are
        object_hit_mask = object_pad & hit_pad

        weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](output, target, mask=object_hit_mask, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses


class RegressionTask(Task):
    def __init__(
        self, 
        name: str, 
        target_labels: list[str], 
        input_object: str, 
        output_object: str, 
        scaler: RegressionTargetScaler, 
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module | None = None,
        add_momentum: bool = False, 
        combine_phi: bool = False,
        **kwargs
    ):   
        """Regression task without uncertainty prediction.

        Parameters
        ----------
        name : str
            Name of the task
        target_labels : list[str]
            List of target names to predict
        input_object : str
            Name of the input object
        output_object : str
            Name of the output object
        scaler : RegressionTargetScaler
            Target scaler object for scaling/unscaling predictions
        losses : dict[str, float]
            Dictionary mapping loss function names to their weights
        costs : dict[str, float]
            Dictionary mapping cost function names to their weights
        net : nn.Module | None
            Neural network module to use. Must not be None.
        add_momentum : bool
            Whether to add scalar momentum to the predictions, computed from the px, py, pz predictions
        """
        super().__init__(**kwargs)
        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_labels = target_labels
        self.scaler = scaler
        self.losses = losses
        self.costs = costs
        self.add_momentum = add_momentum
        self.combine_phi = combine_phi
        if net is None:
            raise ValueError("net parameter must not be None")
        self.net = net
        self.outputs = [self.name]


        if self.add_momentum:
            assert all([t in self.target_labels for t in ["px", "py", "pz"]])
            self.target_labels.append("p")
            self.i_px = self.target_labels.index("px")
            self.i_py = self.target_labels.index("py")
            self.i_pz = self.target_labels.index("pz")
                
        if self.combine_phi:
            assert all([t in self.target_labels for t in ["sinphi", "cosphi"]])
            self.target_labels.append("phi")
            self.i_sinphi = self.target_labels.index("sinphi")
            self.i_cosphi = self.target_labels.index("cosphi")


    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute regression loss between predictions and targets.
        
        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing model outputs. The predictions should be in
            scaled space and have shape [batch_size, num_preds, num_features].
        targets : dict[str, Tensor]
            Dictionary containing target values. The targets will be scaled
            inside this function and should have shape [batch_size, num_targets].
            
        Returns
        -------
        dict[str, dict[str, Tensor]]
            Dictionary mapping loss function names to dictionaries containing:
            - Per-feature losses with keys like 'px_loss', 'py_loss', etc.
            - 'total': weighted sum of all feature losses
        """
        reg_preds = outputs[self.name]  # [batch, num_preds, num_features]

        # Build labels tensor by scaling and concatenating targets
        label_tensors: list[Tensor] = []
        for tgt in self.target_labels:
            try:
                target_key = f"{self.output_object}_{tgt}"
                if target_key not in targets:
                    raise KeyError(f"Missing target {target_key}")
                scaled_tgt = self.scaler.scale(tgt)(targets[target_key])
                label_tensors.append(scaled_tgt.unsqueeze(-1))
            except Exception as e:
                raise RuntimeError(f"Error processing target {tgt}: {str(e)}")
                
        labels = torch.cat(label_tensors, dim=-1)  # [batch, num_targets, num_features]

        # Get valid indices - exclude any target that is all NaN or all zero
        valid_key = f"{self.output_object}_valid"
        if valid_key not in targets:
            raise KeyError(f"Missing validity mask {valid_key}")
            
        valid_idx = targets[valid_key]
        valid_idx = torch.logical_and(valid_idx, ~torch.isnan(labels).all(-1))
        valid_idx = torch.logical_and(valid_idx, ~(labels == 0).all(-1))

        # Initialize nested loss dictionary
        loss_dict: dict[str, dict[str, Tensor]] = {}

        # Calculate losses per loss function
        for loss_fn, loss_weight in self.losses.items():
            # Get the loss function from the registry
            if loss_fn not in loss_fns:
                raise ValueError(f"Unknown loss function {loss_fn}")
                
            loss_fn_callable = loss_fns[loss_fn]

            # Calculate per-feature losses with appropriate reduction
            feature_losses = loss_fn_callable(
                reg_preds[valid_idx],  # [valid_batch, num_features]
                labels[valid_idx],     # [valid_batch, num_features]
            )
            loss_dict[loss_fn] = loss_weight * feature_losses.mean()

        return loss_dict

    # def cost(self, outputs, targets):
        
    #     # Detach predictions and get the relevant predictions
    #     preds = outputs[self.name].detach()
        
    #     # Build labels array just like in loss function
    #     labels = []
    #     for tgt in self.target_labels:
    #         scaled_tgt = self.scaler.scale(tgt)(targets[f"{self.output_object}_{tgt}"])
    #         labels.append(scaled_tgt.unsqueeze(-1))
    #     labels = torch.cat(labels, dim=-1)

    #     # Get valid indices - create a new tensor instead of modifying in place
    #     valid_idx = targets[f"{self.output_object}_valid"].clone()
    #     valid_idx = torch.logical_and(valid_idx, ~torch.isnan(labels).all(-1))
    #     valid_idx = torch.logical_and(valid_idx, ~(labels == 0).all(-1))

    #     # Initialize cost dictionary
    #     cost_dict = {}

    #     for cost_fn, cost_weight in self.costs.items():
    #         # Calculate the base costs between all pred-target pairs
    #         base_costs = cost_weight * cost_fns[cost_fn](preds, labels)
    #         # Create mask for invalid targets
    #         # Create mask for invalid targets - create new tensor
    #         mask = valid_idx.unsqueeze(1).expand(-1, preds.shape[1], -1).clone()
    #         # Create a new tensor for costs with masked values set to 1e6
    #         costs = torch.where(
    #             mask,
    #             base_costs,
    #             torch.full_like(base_costs, 1e6)
    #         )
    #         cost_dict[cost_fn] = costs

    #     # #TODO check if this is correct
    #     # # Expand valid_idx to match [batch, pred, true]
    #     # mask = valid_idx.unsqueeze(1).expand(-1, preds.shape[1], -1)  
    #     # cost_dict[self.name][~mask] = 1e6
        
    #     return cost_dict

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass for regression predictions.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing input tensors. Must contain key f"{input_object}_embed"
            with shape [batch_size, num_queries, embedding_dim].
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing predictions with key self.name and shape
            [batch_size, num_queries, num_features], where num_features is
            len(target_labels).
        """
        input_embed = x[self.input_object + "_embed"]
        # Check input dimensions
        assert input_embed.dim() == 3, f"Expected 3D input tensor, got shape {input_embed.shape}"
        
        # Get predictions from network
        preds = self.net(input_embed)
        
        # Ensure predictions have expected shape
        expected_features = len(self.target_labels) - (1 if self.add_momentum else 0)
        assert preds.shape[-1] == expected_features, \
            f"Expected {expected_features} output features but got {preds.shape[-1]}"
        
        # Add momentum if requested
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)
            
        return {self.name: preds}

    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        preds = torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)
        return preds

    def combine_phi_in_preds(self, preds: Tensor):
        phi =  torch.arctan(preds[..., self.i_sinphi] / preds[..., self.i_cosphi])
        preds = torch.cat([preds, phi.unsqueeze(-1)], dim=-1)
        return preds
    
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Convert raw model outputs to unscaled predictions for evaluation.
        
        This method unscales the predictions since it's used for evaluation and logging,
        not for loss calculation. The loss and cost functions work with scaled values.
        
        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing model outputs with shape [batch_size, num_preds, num_features]
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary mapping each target name to its unscaled predicted values
            in the original target space. Each tensor has shape [batch_size, num_preds].
        """
        if self.name not in outputs:
            raise KeyError(f"Missing predictions for task {self.name}")
            
        preds = outputs[self.name]  # [batch, num_preds, num_features]
        
        if preds.shape[-1] != len(self.target_labels):
            raise ValueError(
                f"Expected predictions with {len(self.target_labels)} features, "
                f"got {preds.shape[-1]}"
            )
        
        # Unscale each feature
        unscaled_preds = {}
        for i, tgt in enumerate(self.target_labels):
            try:
                # All features including momentum need unscaling
                unscaled_preds[f"{self.output_object}_{tgt}"] = self.scaler.inverse(tgt)(preds[..., i])
            except Exception as e:
                raise RuntimeError(f"Error unscaling feature {tgt}: {str(e)}")
                
        return unscaled_preds

class HybridPooling(nn.Module):
    def __init__(self, use_max_only: bool = False):
        super().__init__()
        self.use_max_only = use_max_only
        
    def forward(self, x):
        if self.use_max_only:
            return F.adaptive_max_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.cat([avg_pool, max_pool], dim=1)
    

class InputNormalization(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
    
    def forward(self, x):
        # x shape: [batch, channels, seq_len, features]
        # Permute to [batch, seq_len, channels, features]
        x = x.permute(0, 2, 1, 3)
        # Combine channels and features for normalization
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        # Normalize
        x = self.norm(x)
        # Restore shape and permute back
        x = x.reshape(shape)
        return x.permute(0, 2, 1, 3)


class FPRegressionTaskHitObjRepr(RegressionTask):

    def __init__(
        self,
        name: str,
        target_labels: list[str],
        input_object: str,
        output_object: str,
        scaler: RegressionTargetScaler,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module | None = None,
        add_momentum: bool = False,
        combine_phi: bool = False,
        **kwargs
    ):
        """Regression task that uses hit object representations.
        
        Parameters are the same as RegressionTask, with additional parameters:
        combine_phi : bool
            Whether to combine phi coordinates in the hit representation
        """
        super().__init__(
            name=name,
            target_labels=target_labels,
            input_object=input_object,
            output_object=output_object,
            scaler=scaler,
            losses=losses,
            costs=costs,
            net=net,
            add_momentum=add_momentum,
            combine_phi=combine_phi,
            **kwargs
        )


    def forward(self, x: dict[str, Tensor], track_hit_valid_mask: dict[str, Tensor] | None = None, track_valid_mask: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass for regression predictions using hit object representations.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing input tensors. Must contain:
            - query_embed: [batch_size, num_queries, embedding_dim]
            - hit_embed: [batch_size, num_hits, embedding_dim]
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing predictions with key self.name and shape
            [batch_size, num_queries, num_features]
        """
        query_embed = x["query_embed"]
        hit_embed = x["hit_embed"]

        if track_hit_valid_mask is None:
            raise ValueError("track_hit_valid_mask is required")
        
        is_mask = isinstance(track_hit_valid_mask, dict)
        
        def get_hit_embeds_from_logits(track_hit_assignment_logits, hit_embed):
            # Count number of hits per track using a threshold on the logits

            track_hit_assignment_mask = (track_hit_assignment_logits > 0.1)
            num_hits_per_track = track_hit_assignment_mask.sum(dim=-1)  # [batch, num_queries]
            
            # Initialize output tensor
            batch_size, num_queries, _ = track_hit_assignment_logits.shape
            track_hit_embeds = torch.zeros(batch_size, num_queries, hit_embed.shape[-1], device=hit_embed.device)

            # For each track that has hits assigned, gather and do weighted average of its hit embeddings
            for b in range(batch_size):
                for q in range(num_queries):
                    if num_hits_per_track[b, q] > 0:
                        # Get indices and scores of assigned hits
                        hit_indices = torch.where(track_hit_assignment_mask[b, q])[0]
                        
                        hit_scores = track_hit_assignment_logits[b, q, hit_indices].sigmoid()  # Convert logits to probabilities
                        
                        # Gather coordinates for selected hits
                        track_hits = hit_embed[b, hit_indices]  # [num_hits, embed_dim]
                        
                        # Compute weighted average using the scores
                        weights = hit_scores.unsqueeze(-1)  # [num_hits, 1]
                        weighted_sum = (track_hits * weights).sum(dim=0)  # [embed_dim]
                        normalization = weights.sum()

                        # Store normalized weighted average
                        track_hit_embeds[b, q] = weighted_sum / normalization

            return track_hit_embeds
        
        def get_hit_embeds(track_hit_assignment: Tensor, hit_embed: Tensor) -> Tensor:
            # Count number of hits per track
            num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]
            
            # Initialize output tensor
            batch_size, num_queries, _ = track_hit_assignment.shape
            track_hit_embeds = torch.zeros(batch_size, num_queries, hit_embed.shape[-1], device=hit_embed.device)
            
            # For each track that has hits assigned, gather and average its hit embeddings
            for b in range(batch_size):
                for q in range(num_queries):
                    if num_hits_per_track[b, q] > 0:
                        # Get indices of assigned hits
                        hit_indices = torch.where(track_hit_assignment[b, q])[0]
                        
                        # Gather embeddings for all assigned hits
                        track_hits = hit_embed[b, hit_indices]  # [num_hits, embed_dim]
                        
                        # Average the embeddings
                        track_hit_embeds[b, q] = track_hits.mean(dim=0)

            return track_hit_embeds


        # Get hit embeddings based on track-hit assignments
        if is_mask:
            track_hit_assignment = track_hit_valid_mask["track_hit_valid"]  # [batch, num_queries, num_hits]
            track_hit_embeds = get_hit_embeds(track_hit_assignment, hit_embed)
        else:
            track_hit_assignment_logits = track_hit_valid_mask
            track_hit_embeds = get_hit_embeds_from_logits(track_hit_assignment_logits, hit_embed)

        # Combine query embeddings with hit embeddings
        combined_embed = torch.cat([query_embed, track_hit_embeds], dim=-1)
        
        # Get predictions from network
        preds = self.net(combined_embed)
        
        # Ensure predictions have expected shape
        expected_features = len(self.target_labels) - (1 if self.add_momentum else 0)
        assert preds.shape[-1] == expected_features, \
            f"Expected {expected_features} output features but got {preds.shape[-1]}"
        
        # Add momentum if requested
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)
            
        return {self.name: preds}






class FPRegressionTaskHitObjCNN(RegressionTask):

    def __init__(
        self,
        name: str,
        target_labels: list[str],
        input_object: str,
        output_object: str,
        scaler: RegressionTargetScaler,
        losses: dict[str, float],
        costs: dict[str, float],
        net: nn.Module | None = None,
        add_momentum: bool = False,
        combine_phi: bool = False,
        max_hits: int = 20,
        cnn_hidden_dim: int = 64,
        cnn_out_dim: int = 64,
        dropout_rate: float = 0.2,
        num_features: int = 6,
        use_max_only: bool = False,
        add_positional_features: bool = False,
        plot_dir: str = "plots/",
        **kwargs
    ):
        """Regression task that uses hit object representations.
        
        Parameters are the same as RegressionTask, with additional parameters:
        combine_phi : bool
            Whether to combine phi coordinates in the hit representation
        """
        super().__init__(
            name=name,
            target_labels=target_labels,
            input_object=input_object,
            output_object=output_object,
            scaler=scaler,
            losses=losses,
            costs=costs,
            net=net,
            add_momentum=add_momentum,
            combine_phi=combine_phi,
            **kwargs
        )

        self.max_hits = max_hits
        self.cnn_hidden_dim = cnn_hidden_dim
        self.cnn_out_dim = cnn_out_dim
        self.use_max_only = use_max_only
        self.add_positional_features = add_positional_features
        if add_positional_features:
            num_features += 1
        self.num_features = num_features
        self.plot_dir = plot_dir
        self.step = 0

        # Create plot directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)

        # CNN for processing hit coordinates
        # Input shape: [batch * num_tracks, 1, max_hits, num_coords + 1]  # for pos encoding
        self.hit_cnn = nn.Sequential(
            # Input normalization
            InputNormalization(num_features=num_features),  # n coords + pos_encoding
            
            # First conv block
            nn.Conv2d(1, cnn_hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.LayerNorm([cnn_hidden_dim, max_hits, num_features]),
            
            # Second conv block
            nn.Conv2d(cnn_hidden_dim, cnn_hidden_dim * 2, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
            nn.GroupNorm(num_groups=8, num_channels=cnn_hidden_dim*2),
            
            # Pooling
            HybridPooling(use_max_only=use_max_only),
            nn.Flatten(),
        )


    def add_positional_features(self, hits: Tensor) -> Tensor:
        """Add radius and normalized position features to hit coordinates.
        
        Parameters
        ----------
        hits : Tensor [batch, num_hits, 3]
            Hit coordinates (x, y, z)
            
        Returns
        -------
        Tensor [batch, num_hits, 4]
            Hit coordinates with added normalized position
        """
        
        # Create normalized position encoding
        batch_size, num_hits = hits.shape[:2]
        pos_encoding = torch.arange(num_hits, device=hits.device).float()
        pos_encoding = pos_encoding / (num_hits - 1)  # Normalize to [0, 1]
        pos_encoding = pos_encoding.view(1, -1, 1).expand(batch_size, -1, -1)
        
        # Concatenate original coordinates with r and position encoding
        return torch.cat([hits, pos_encoding], dim=-1)

    def plot_hit_groups(self, batch_idx: int, hit_coords: torch.Tensor, hit_groups: list[torch.Tensor], scores: list[torch.Tensor]):
        """Plot hit coordinates for each group in different colors.
        
        Parameters
        ----------
        batch_idx : int
            Index of the batch to plot
        hit_coords : torch.Tensor
            Full tensor of hit coordinates [num_hits, num_features]
        hit_groups : list[torch.Tensor] 
            List of tensors containing indices for each group's hits
        scores : list[torch.Tensor]
            List of tensors containing scores for each group's hits
        """
        plt.figure(figsize=(10, 10))
        
        # Generate a color for each group
        colors = plt.cm.rainbow(np.linspace(0, 1, len(hit_groups)))
        
        # Plot each group with a different color
        for group_idx, (group_indices, group_scores) in enumerate(zip(hit_groups, scores)):
            if len(group_indices) == 0:
                continue
                
            # Get coordinates for this group
            group_coords = hit_coords[group_indices]
            x_coords = group_coords[:, 0].cpu().numpy()
            y_coords = group_coords[:, 1].cpu().numpy()
            
            # Convert scores to numpy and normalize for alpha values
            alpha_values = group_scores.to(torch.float32).sigmoid().cpu().numpy()

            
            # Plot points with alpha values based on scores
            plt.scatter(x_coords, y_coords, c=[colors[group_idx]], alpha=alpha_values, 
                       label=f'Group {group_idx}')
        
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.title(f'Hit Groups - Event {batch_idx}, Step {self.step}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.plot_dir, f'hits_event{batch_idx}_step{self.step}_{timestamp}.png'))
        plt.close()
        
        self.step += 1

    def get_hit_embeds_from_logits(self, track_hit_assignment_logits, hit_embed):

        # Convert logits to probabilities once
        hit_scores = track_hit_assignment_logits.sigmoid()  # [batch, num_queries, num_hits]
        
        # Create mask for hits above threshold
        track_hit_assignment_mask = (hit_scores > 0.1)  # [batch, num_queries, num_hits]
        num_hits_per_track = track_hit_assignment_mask.sum(dim=-1)  # [batch, num_queries]
        
        # Initialize output tensor
        batch_size, num_queries, _ = track_hit_assignment_logits.shape
        embeds_shape = self.cnn_hidden_dim if self.use_max_only else 4 * self.cnn_hidden_dim
        track_hit_embeds = torch.zeros(batch_size, num_queries, embeds_shape, device=hit_embed.device)

        plot_batch_idx = torch.randint(0, batch_size, (1,)).item()

        # For each track that has hits assigned, gather and do weighted average of its hit embeddings
        for b in range(batch_size):
            do_plot = True if (b==plot_batch_idx) else False
            hit_groups = []
            hit_scores = []
            for q in range(num_queries):
                if num_hits_per_track[b, q] > 0:
                    # Get indices of hits above threshold
                    hit_indices = torch.where(track_hit_assignment_mask[b, q])[0]
                    # Get scores for valid hits
                    valid_hit_scores = track_hit_assignment_logits[b, q, hit_indices]

                     # If we have more hits than max_hits, take top scoring ones
                    if len(hit_indices) > self.max_hits:
                        # Get indices of top scoring hits
                        _, top_k_indices = torch.topk(valid_hit_scores, k=self.max_hits)
                        selected_indices = hit_indices[top_k_indices]
                        valid_hit_scores = valid_hit_scores[top_k_indices]
                    else:
                        selected_indices = hit_indices

                    if do_plot:
                        # Store for visualization
                        hit_groups.append(selected_indices)
                        hit_scores.append(valid_hit_scores)

                    # Get embeddings for selected hits
                    selected_hits = hit_embed[b, selected_indices]  # [num_selected, embed_dim]

                    # Sort by radius if available (assuming last dimension is radius)
                    if selected_hits.size(1) > 0:  # Check if we have any hits
                        r = selected_hits[:, -1]  # Get radius values
                        _, r_indices = torch.sort(r)
                        selected_hits = selected_hits[r_indices]

                    #TODO Have a look at how many hits each track has

                    # Add positional features if needed
                    if hasattr(self, 'add_positional_features') and self.add_positional_features:
                        selected_hits = self.add_positional_features(selected_hits.unsqueeze(0))[0]
                    
                    # Pad if necessary
                    if len(selected_hits) < self.max_hits:
                        padding = torch.zeros(
                            self.max_hits - len(selected_hits),
                            selected_hits.size(1),
                            device=selected_hits.device
                        )
                        padded_hits = torch.cat([selected_hits, padding], dim=0)
                    else:
                        padded_hits = selected_hits

                    # Add batch and channel dimensions for CNN
                    cnn_input = padded_hits.unsqueeze(0).unsqueeze(0)  # [1, 1, max_hits, num_coords]
                    
                    # Apply CNN
                    hit_features = self.hit_cnn(cnn_input)  # [1, cnn_hidden_dim * 2]

                    track_hit_embeds[b, q] = hit_features.squeeze(0)
                        # Plot hits for this batch
            if (len(hit_groups) > 0) and (do_plot):
                self.plot_hit_groups(b, hit_embed[b], hit_groups, hit_scores)

        return track_hit_embeds


    def forward(self, x: dict[str, Tensor], track_hit_valid_mask: dict[str, Tensor] | None = None, track_valid_mask: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass for regression predictions using hit object representations.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing input tensors. Must contain:
            - query_embed: [batch_size, num_queries, embedding_dim]
            - hit_embed: [batch_size, num_hits, embedding_dim]
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing predictions with key self.name and shape
            [batch_size, num_queries, num_features]
        """
        query_embed = x["query_embed"]
        hit_embed = x["hit_coords"]

        if track_hit_valid_mask is None:
            raise ValueError("track_hit_valid_mask is required")
        
        track_hit_assignment_logits = track_hit_valid_mask["track_hit_valid"]  # [batch, num_queries, num_hits]
        assert track_hit_assignment_logits.dtype in [torch.float32, torch.float64, torch.bfloat16], f"Expected float mask, got {track_hit_assignment_logits.dtype}"

        track_hit_embeds = self.get_hit_embeds_from_logits(track_hit_assignment_logits, hit_embed)

        # Combine query embeddings with hit embeddings
        combined_embed = torch.cat([query_embed, track_hit_embeds], dim=-1)
        
        # Get predictions from network
        preds = self.net(combined_embed)
        
        # Ensure predictions have expected shape
        expected_features = len(self.target_labels) - (1 if self.add_momentum else 0)
        assert preds.shape[-1] == expected_features, \
            f"Expected {expected_features} output features but got {preds.shape[-1]}"
        
        # Add momentum if requested
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)
            
        return {self.name: preds}



class FPRegressionTask(Task):
    def __init__(
        self, 
        name: str, 
        target_labels: list[str], 
        input_object: str, 
        add_embed: str | None,
        output_object: str, 
        scaler: RegressionTargetScaler, 
        losses: dict[str, float],
        costs: dict[str, float],
        hit_input_vars: list[str],
        max_hits: int,
        net: nn.Module | None = None,
        add_momentum: bool = False, 
        combine_phi: bool = False,
        **kwargs
    ):   
        """Regression task without uncertainty prediction.

        Parameters
        ----------
        name : str
            Name of the task
        target_labels : list[str]
            List of target names to predict
        input_object : str
            Name of the input object
        output_object : str
            Name of the output object
        scaler : RegressionTargetScaler
            Target scaler object for scaling/unscaling predictions
        losses : dict[str, float]
            Dictionary mapping loss function names to their weights
        costs : dict[str, float]
            Dictionary mapping cost function names to their weights
        net : nn.Module | None
            Neural network module to use. Must not be None.
        add_momentum : bool
            Whether to add scalar momentum to the predictions, computed from the px, py, pz predictions
        """
        super().__init__(**kwargs)
        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_labels = target_labels
        self.scaler = scaler
        self.losses = losses
        self.costs = costs
        self.add_embed = add_embed
        self.add_momentum = add_momentum
        self.combine_phi = combine_phi
        self.hit_input_vars = hit_input_vars
        self.max_hits = max_hits
        
        # Network to process query embeddings
        if net is None:
            raise ValueError("net parameter must not be None")
        self.net = net
        
        # Output dimension for predictions
        self.output_dim = len(self.target_labels) + (1 if self.add_momentum else 0) + (1 if self.combine_phi else 0)
        
        self.outputs = [self.name]

        if self.add_momentum:
            assert all([t in self.target_labels for t in ["px", "py", "pz"]]), "px, py, pz required for momentum calculation"
            self.target_labels.append("p")
            self.i_px = self.target_labels.index("px")
            self.i_py = self.target_labels.index("py")
            self.i_pz = self.target_labels.index("pz")

                
        if self.combine_phi:
            assert all([t in self.target_labels for t in ["sinphi", "cosphi"]])
            self.target_labels.append("phi")
            self.i_sinphi = self.target_labels.index("sinphi")
            self.i_cosphi = self.target_labels.index("cosphi")

    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute regression loss between predictions and targets.
        
        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing model outputs. The predictions should be in
            scaled space and have shape [batch_size, num_preds, num_features].
        targets : dict[str, Tensor]
            Dictionary containing target values. The targets will be scaled
            inside this function and should have shape [batch_size, num_targets].
            
        Returns
        -------
        dict[str, dict[str, Tensor]]
            Dictionary mapping loss function names to dictionaries containing:
            - Per-feature losses with keys like 'px_loss', 'py_loss', etc.
            - 'total': weighted sum of all feature losses
        """
        reg_preds = outputs[self.name]  # [batch, num_preds, num_features]

        # Build labels tensor by scaling and concatenating targets
        label_tensors: list[Tensor] = []
        for tgt in self.target_labels:
            try:
                target_key = f"{self.output_object}_{tgt}"
                if target_key not in targets:
                    raise KeyError(f"Missing target {target_key}")
                scaled_tgt = self.scaler.scale(tgt)(targets[target_key])
                label_tensors.append(scaled_tgt.unsqueeze(-1))
            except Exception as e:
                raise RuntimeError(f"Error processing target {tgt}: {str(e)}")
                
        labels = torch.cat(label_tensors, dim=-1)  # [batch, num_targets, num_features]

        # Get valid indices - exclude any target that is all NaN or all zero
        valid_key = f"{self.output_object}_valid"
        if valid_key not in targets:
            raise KeyError(f"Missing validity mask {valid_key}")
            
        valid_idx = targets[valid_key]
        valid_idx = torch.logical_and(valid_idx, ~torch.isnan(labels).all(-1))
        valid_idx = torch.logical_and(valid_idx, ~(labels == 0).all(-1))

        # Initialize nested loss dictionary
        loss_dict: dict[str, dict[str, Tensor]] = {}

        # Calculate losses per loss function
        for loss_fn, loss_weight in self.losses.items():
            # Get the loss function from the registry
            if loss_fn not in loss_fns:
                raise ValueError(f"Unknown loss function {loss_fn}")
                
            loss_fn_callable = loss_fns[loss_fn]
            
            # Skip calculation if no valid samples
            if not valid_idx.any():
                feature_losses = torch.zeros(len(self.target_labels), device=reg_preds.device)
            else:
                # Calculate per-feature losses with appropriate reduction
                feature_losses = loss_fn_callable(
                    reg_preds[valid_idx],  # [valid_batch, num_features]
                    labels[valid_idx],     # [valid_batch, num_features]
                )
            
            # # Store per-feature losses
            # for i, tgt in enumerate(self.target_labels):
            #     loss_dict[loss_fn][f"{tgt}_loss"] = loss_weight * feature_losses[i]
                
            # Store total loss
            loss_dict[loss_fn] = loss_weight * feature_losses.mean()

        return loss_dict

    def cost(self, outputs, targets):
        
        # Detach predictions and get the relevant predictions
        preds = outputs[self.name].detach()
        
        # Build labels array just like in loss function
        labels = []
        for tgt in self.target_labels:
            scaled_tgt = self.scaler.scale(tgt)(targets[f"{self.output_object}_{tgt}"])
            labels.append(scaled_tgt.unsqueeze(-1))
        labels = torch.cat(labels, dim=-1)

        # Get valid indices - create a new tensor instead of modifying in place
        valid_idx = targets[f"{self.output_object}_valid"].clone()
        valid_idx = torch.logical_and(valid_idx, ~torch.isnan(labels).all(-1))
        valid_idx = torch.logical_and(valid_idx, ~(labels == 0).all(-1))

        # Initialize cost dictionary
        cost_dict = {}

        for cost_fn, cost_weight in self.costs.items():
            # Calculate the base costs between all pred-target pairs
            base_costs = cost_weight * cost_fns[cost_fn](preds, labels)
            # Create mask for invalid targets
            mask = valid_idx.unsqueeze(1).expand(-1, preds.shape[1], -1).clone()
            # Create a new tensor for costs with masked values set to 1e6
            costs = torch.where(
                mask,
                base_costs,
                torch.full_like(base_costs, 1e6)
            )
            cost_dict[cost_fn] = costs
        
        return cost_dict

    def forward(self, x: dict[str, Tensor], track_hit_valid_mask: dict[str, Tensor] | None = None, track_valid_mask: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass for regression predictions.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing input tensors. Must contain key f"{input_object}_embed"
            with shape [batch_size, num_queries, embedding_dim].
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing predictions with key self.name and shape
            [batch_size, num_queries, num_features], where num_features is
            len(target_labels).
        """
        input_embed = x["query_embed"]
        # hit_valid_mask = track_valid_mask["hit_valid"]
        #TODO only 6 hits?

        # want tensor of shape [batch, num_queries, 3*max_num_hits]
        # where max_num_hits is the max number of hits per track
        # and the tensor is filled with the coords of the hits that are assigned to the tracks
        # and padded with zeros for the rest
        # if no track hit assignment mask is provided, then throw error

        input_coords = x["hit_coords"]  # [batch, hits, 3]
        
        # Get track-hit assignments, error if not provided
        if track_hit_valid_mask is None:
            raise ValueError("track_hit_valid_mask is required")
        
        is_mask = isinstance(track_hit_valid_mask, dict)


        def get_hit_coords(track_hit_assignment: Tensor, input_coords: Tensor) -> Tensor:
            """Process hit coordinates from binary assignment mask.
            
            Parameters
            ----------
            track_hit_assignment : Tensor [batch, num_queries, num_hits]
                Binary mask indicating which hits belong to which track
            input_coords : Tensor [batch, num_hits, coord_dim]
                Input coordinates for each hit
                
            Returns
            -------
            Tensor [batch, num_queries, max_hits * coord_dim]
                Flattened and padded hit coordinates for each track
            """
            batch_size, num_queries, num_hits = track_hit_assignment.shape
            coord_dim = input_coords.shape[-1]
            
            # Initialize output tensor
            track_hit_coords = torch.zeros(
                batch_size, num_queries, self.max_hits * coord_dim,
                device=input_coords.device
            )

            # Process each batch and query
            for b in range(batch_size):
                for q in range(num_queries):
                    # Get indices of assigned hits for this track
                    hit_indices = torch.where(track_hit_assignment[b, q])[0]
                    
                    if len(hit_indices) == 0:
                        continue
                        
                    # If we have more hits than max_hits, randomly select max_hits
                    if len(hit_indices) > self.max_hits:
                        perm = torch.randperm(len(hit_indices), device=hit_indices.device)
                        hit_indices = hit_indices[perm[:self.max_hits]]
                    
                    # Get coordinates for selected hits
                    selected_coords = input_coords[b, hit_indices]  # [num_selected, coord_dim]
                    
                    # Flatten coordinates and store in output tensor
                    flat_coords = selected_coords.reshape(-1)  # [num_selected * coord_dim]
                    track_hit_coords[b, q, :len(flat_coords)] = flat_coords

            return track_hit_coords
        
        def get_hit_coords_from_logits(track_hit_assignment_logits: Tensor, input_coords: Tensor) -> Tensor:
            """Process hit coordinates from assignment logits.
            
            Parameters
            ----------
            track_hit_assignment_logits : Tensor [batch, num_queries, num_hits]
                Logits indicating hit assignment probabilities
            input_coords : Tensor [batch, num_hits, coord_dim]
                Input coordinates for each hit
                
            Returns
            -------
            Tensor [batch, num_queries, max_hits * coord_dim]
                Flattened and padded hit coordinates for each track
            """
            batch_size, num_queries, num_hits = track_hit_assignment_logits.shape
            coord_dim = input_coords.shape[-1]
            
            # Initialize output tensor
            track_hit_coords = torch.zeros(
                batch_size, num_queries, self.max_hits * coord_dim,
                device=input_coords.device
            )

            # Convert logits to probabilities
            probs = track_hit_assignment_logits.sigmoid()

            # Process each batch and query
            for b in range(batch_size):
                for q in range(num_queries):
                    # Get hits above threshold
                    valid_hits = torch.where(probs[b, q] > 0.1)[0]
                    
                    if len(valid_hits) == 0:
                        continue
                    
                    # If we have more hits than max_hits, take top scoring ones
                    if len(valid_hits) > self.max_hits:
                        scores = probs[b, q, valid_hits]
                        _, top_indices = torch.topk(scores, k=self.max_hits)
                        valid_hits = valid_hits[top_indices]
                    
                    # Get coordinates for selected hits
                    selected_coords = input_coords[b, valid_hits]  # [num_selected, coord_dim]
                    
                    # Flatten coordinates and store in output tensor
                    flat_coords = selected_coords.reshape(-1)  # [num_selected * coord_dim]
                    track_hit_coords[b, q, :len(flat_coords)] = flat_coords

            return track_hit_coords



        # def get_hit_coords(track_hit_assignment: Tensor, input_coords: Tensor) -> Tensor:
        #     """Vectorized version of hit coordinate processing.
            
        #     Parameters
        #     ----------
        #     track_hit_assignment : Tensor [batch, num_queries, num_hits]
        #         Binary mask indicating which hits belong to which track
            # input_coords : Tensor [batch, num_hits, coord_dim]
            #     Input coordinates for each hit
                
            # Returns
            # -------
            # Tensor [batch, num_queries, max_hits * coord_dim]
            #     Flattened and padded hit coordinates for each track
            # """
            # batch_size, num_queries, num_hits = track_hit_assignment.shape
            # coord_dim = input_coords.shape[-1]
            
            # # Count hits per track
            # num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]
            
            # # Create output tensor
            # track_hit_coords = torch.zeros(
            #     batch_size, num_queries, self.max_hits * coord_dim,
            #     device=input_coords.device
            # )

            # # For tracks with hits, gather and process their coordinates
            # has_hits = num_hits_per_track > 0
            # if not has_hits.any():
            #     return track_hit_coords

            # # Get indices of hits for each track
            # hit_indices = torch.nonzero(track_hit_assignment.view(-1, num_hits), as_tuple=False)
            # batch_query_idx = hit_indices[:, 0]
            # hit_idx = hit_indices[:, 1]

            # # Reshape batch_query_idx to get batch and query indices
            # batch_idx = batch_query_idx // num_queries
            # query_idx = batch_query_idx % num_queries

            # # Gather coordinates for all hits
            # gathered_coords = input_coords[batch_idx, hit_idx]  # [num_total_hits, coord_dim]

            # # Create position indices for scattering
            # hits_so_far = torch.zeros(batch_size, num_queries, dtype=torch.long, device=input_coords.device)
            # pos_indices = hits_so_far[batch_idx, query_idx]
            # hits_so_far[batch_idx, query_idx] += 1

            # # Only keep up to max_hits coordinates per track
            # valid_hits = pos_indices < self.max_hits
            # if valid_hits.any():
            #     batch_idx = batch_idx[valid_hits]
            #     query_idx = query_idx[valid_hits]
            #     pos_indices = pos_indices[valid_hits]
            #     gathered_coords = gathered_coords[valid_hits]

            #     # Calculate indices for flattened coordinates
            #     flat_coords = gathered_coords.view(-1)  # [num_valid_hits * coord_dim]
            #     coord_indices = pos_indices.unsqueeze(-1) * coord_dim + torch.arange(coord_dim, device=input_coords.device)
            #     coord_indices = coord_indices.view(-1)  # [num_valid_hits * coord_dim]

            #     # Scatter the coordinates into the output tensor
            #     track_hit_coords[batch_idx, query_idx, coord_indices] = flat_coords

            # return track_hit_coords
        
        # def get_hit_coords_from_logits(track_hit_assignment_logits: Tensor, input_coords: Tensor) -> Tensor:
        #     """Vectorized version of hit coordinate processing from logits.
            
        #     Parameters
        #     ----------
        #     track_hit_assignment_logits : Tensor [batch, num_queries, num_hits]
        #         Logits indicating hit assignment probabilities
        #     input_coords : Tensor [batch, num_hits, coord_dim]
        #         Input coordinates for each hit
                
        #     Returns
        #     -------
        #     Tensor [batch, num_queries, max_hits * coord_dim]
        #         Flattened and padded hit coordinates for each track
        #     """
        #     # Convert logits to mask using threshold
        #     track_hit_assignment = (track_hit_assignment_logits > 0.1)
            
        #     # For tracks with more hits than max_hits, keep only top scoring hits
        #     num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]
        #     too_many_hits = num_hits_per_track > self.max_hits
            
        #     if too_many_hits.any():
        #         # Get scores for assigned hits
        #         scores = track_hit_assignment_logits.masked_fill(~track_hit_assignment, float('-inf'))
        #         # Get top k scores
        #         _, top_indices = torch.topk(scores, k=self.max_hits, dim=-1)
        #         # Create new assignment mask with only top k hits
        #         new_assignment = torch.zeros_like(track_hit_assignment)
        #         new_assignment.scatter_(-1, top_indices, 1)
        #         # Update assignment mask
        #         track_hit_assignment = torch.where(too_many_hits.unsqueeze(-1), new_assignment, track_hit_assignment)
            
        #     return get_hit_coords(track_hit_assignment, input_coords)
        
        # def get_hit_coords_from_logits(track_hit_assignment_logits, input_coords):
        #     # Count number of hits per track
        #     track_hit_assignment_mask = (track_hit_assignment_logits > 0.1)
        #     num_hits_per_track = track_hit_assignment_mask.sum(dim=-1)  # [batch, num_queries]
            
        #     # Initialize output tensor
        #     batch_size, num_queries, _ = track_hit_assignment_logits.shape
        #     track_hit_coords = torch.zeros(batch_size, num_queries, self.max_hits * input_coords.shape[-1], device=input_coords.device)
            
        #     # For each track that has hits assigned, gather and flatten its hit coordinates
        #     for b in range(batch_size):
        #         for q in range(num_queries):
        #             if num_hits_per_track[b, q] > 0:
        #                 # Get indices of assigned hits
        #                 hit_indices = torch.where(track_hit_assignment_mask[b, q])[0]
                        
        #                 # If we have more hits than max_hits, select the top scoring hits
        #                 if len(hit_indices) > self.max_hits:
        #                     # Get the logit scores for the assigned hits
        #                     hit_scores = track_hit_assignment_logits[b, q, hit_indices]
        #                     # Sort hits by scores in descending order and get indices
        #                     _, top_k_indices = torch.topk(hit_scores, k=self.max_hits)
        #                     hit_indices = hit_indices[top_k_indices]
                        
        #                 # Gather coordinates for selected hits
        #                 track_hits = input_coords[b, hit_indices]  # [num_hits, 3]
                        
        #                 # Handle padding - only take up to max_hits coordinates
        #                 num_hits_to_use = min(len(hit_indices), self.max_hits)
        #                 flattened_coords = track_hits[:num_hits_to_use].flatten()  # [num_hits_to_use * 3]
                        
        #                 # Store coordinates - the rest will remain as zeros for padding
        #                 track_hit_coords[b, q, :num_hits_to_use * 3] = flattened_coords

        #     return track_hit_coords
        
        # def get_hit_coords(track_hit_assignment, input_coords):
        #     # Count number of hits per track
        #     num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]
            
        #     # Initialize output tensor
        #     batch_size, num_queries, num_input_hit_vars = track_hit_assignment.shape
        #     track_hit_coords = torch.zeros(batch_size, num_queries, self.max_hits * input_coords.shape[-1], device=input_coords.device)
            
        #     # For each track that has hits assigned, gather and flatten its hit coordinates
        #     for b in range(batch_size):
        #         for q in range(num_queries):
        #             if num_hits_per_track[b, q] > 0:
        #                 # Get indices of assigned hits
        #                 hit_indices = torch.where(track_hit_assignment[b, q])[0]
                        
        #                 # If we have more hits than max_hits, randomly select max_hits of them
        #                 if len(hit_indices) > self.max_hits:
        #                     perm = torch.randperm(len(hit_indices), device=hit_indices.device)
        #                     hit_indices = hit_indices[perm[:self.max_hits]]
                        
        #                 # Gather coordinates for selected hits
        #                 track_hits = input_coords[b, hit_indices]  # [num_hits, 3]
                        
        #                 # Handle padding - only take up to max_hits coordinates
        #                 num_hits_to_use = min(len(hit_indices), self.max_hits)
        #                 flattened_coords = track_hits[:num_hits_to_use].flatten()  # [num_hits_to_use * 3]
                        
        #                 # Store coordinates - the rest will remain as zeros for padding
        #                 track_hit_coords[b, q, :num_hits_to_use * 3] = flattened_coords

        #     return track_hit_coords

        if is_mask:
            track_hit_assignment = track_hit_valid_mask["track_hit_valid"]  # [batch, num_queries, num_hits]
            # Get hit coordinates
            track_hit_coords = get_hit_coords(track_hit_assignment, input_coords)
        else:
            track_hit_assignment_logits = track_hit_valid_mask.detach().sigmoid()
            track_hit_coords = get_hit_coords_from_logits(track_hit_assignment_logits, input_coords)


            # Now track_hit_coords has shape [batch, num_queries, max_hits*3]
            # with coordinates of assigned hits, padded with zeros

        # Check input dimensions
        assert input_embed.dim() == 3, f"Expected 3D input tensor, got shape {input_embed.shape}"

        embed = torch.cat([input_embed, track_hit_coords], dim=-1)
        
        # Get predictions from network
        preds = self.net(embed)
        # preds = self.net(input_embed)

        # Add momentum if requested
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)

        if self.combine_phi:
            preds = self.combine_phi_in_preds(preds)
            
        # Ensure predictions have expected shape
        assert preds.shape[-1] == self.output_dim, \
            f"Expected {self.output_dim} output features but got {preds.shape[-1]}"
            
        return {self.name: preds}


    def add_momentum_to_preds(self, preds: Tensor):
        momentum = torch.sqrt(preds[..., self.i_px] ** 2 + preds[..., self.i_py] ** 2 + preds[..., self.i_pz] ** 2)
        preds = torch.cat([preds, momentum.unsqueeze(-1)], dim=-1)
        return preds
    
    def combine_phi_in_preds(self, preds: Tensor):
        phi =  torch.arctan(preds[..., self.i_sinphi] / preds[..., self.i_cosphi])
        preds = torch.cat([preds, phi.unsqueeze(-1)], dim=-1)
        return preds

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Convert raw model outputs to unscaled predictions for evaluation.
        
        This method unscales the predictions since it's used for evaluation and logging,
        not for loss calculation. The loss and cost functions work with scaled values.
        
        Parameters
        ----------
        outputs : dict[str, Tensor]
            Dictionary containing model outputs with shape [batch_size, num_preds, num_features]
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary mapping each target name to its unscaled predicted values
            in the original target space. Each tensor has shape [batch_size, num_preds].
        """
        if self.name not in outputs:
            raise KeyError(f"Missing predictions for task {self.name}")
            
        preds = outputs[self.name]  # [batch, num_preds, num_features]
        
        if preds.shape[-1] != len(self.target_labels):
            raise ValueError(
                f"{self.target_labels}"
                f"Expected predictions with {len(self.target_labels)} features, "
                f"got {preds.shape[-1]}"
            )
        
        # Unscale each feature
        unscaled_preds = {}
        for i, tgt in enumerate(self.target_labels):
            try:
                # All features including momentum need unscaling
                unscaled_preds[f"{self.output_object}_{tgt}"] = self.scaler.inverse(tgt)(preds[..., i])
            except Exception as e:
                raise RuntimeError(f"Error unscaling feature {tgt}: {str(e)}")
                
        return unscaled_preds





class FPRegressionTaskHitOnly(FPRegressionTask):

       
    def cost(self, outputs, targets):
        
        # Detach predictions and get the relevant predictions
        preds = outputs[self.name].detach()
        
        # Build labels array just like in loss function
        labels = []
        for tgt in self.target_labels:
            scaled_tgt = self.scaler.scale(tgt)(targets[f"{self.output_object}_{tgt}"])
            scaled_tgt = torch.nan_to_num(scaled_tgt, nan=0.0, posinf=1e4, neginf=-1e4)
            labels.append(scaled_tgt.unsqueeze(-1))
        labels = torch.cat(labels, dim=-1)

        # Get valid indices - create a new tensor instead of modifying in place
        valid_idx = targets[f"{self.output_object}_valid"].clone()
        valid_idx = torch.logical_and(valid_idx, ~torch.isnan(labels).all(-1))
        valid_idx = torch.logical_and(valid_idx, ~(labels == 0).all(-1))

        # Initialize cost dictionary
        cost_dict = {}

        for cost_fn, cost_weight in self.costs.items():
            # Calculate the base costs between all pred-target pairs
            base_costs = cost_weight * cost_fns[cost_fn](preds, labels)
            # Create mask for invalid targets
            # Create mask for invalid targets - create new tensor
            mask = valid_idx.unsqueeze(1).expand(-1, preds.shape[1], -1).clone()
            # Create a new tensor for costs with masked values set to 1e6
            costs = torch.where(
                mask,
                base_costs,
                torch.full_like(base_costs, 1e6)
            )
            cost_dict[cost_fn] = costs

        # #TODO check if this is correct
        # # Expand valid_idx to match [batch, pred, true]
        # mask = valid_idx.unsqueeze(1).expand(-1, preds.shape[1], -1)  
        # cost_dict[self.name][~mask] = 1e6
        
        return cost_dict

    def forward(self, x: dict[str, Tensor], track_hit_valid_mask: dict[str, Tensor] | None = None, track_valid_mask: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass for regression predictions.
        
        Parameters
        ----------
        x : dict[str, Tensor]
            Dictionary containing input tensors. Must contain key f"{input_object}_embed"
            with shape [batch_size, num_queries, embedding_dim].
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing predictions with key self.name and shape
            [batch_size, num_queries, num_features], where num_features is
            len(target_labels).
        """
        input_embed = x["query_embed"]
        # hit_valid_mask = track_valid_mask["hit_valid"]
        #TODO only 6 hits?

        # want tensor of shape [batch, num_queries, 3*max_num_hits]
        # where max_num_hits is the max number of hits per track
        # and the tensor is filled with the coords of the hits that are assigned to the tracks
        # and padded with zeros for the rest
        # if no track hit assignment mask is provided, then throw error

        input_coords = x["hit_coords"]  # [batch, hits, 3]
        
        # Get track-hit assignments, error if not provided
        if track_hit_valid_mask is None:
            raise ValueError("track_hit_valid_mask is required")
        track_hit_assignment = track_hit_valid_mask["track_hit_valid"]  # [batch, num_queries, num_hits]

        def get_hit_coords(track_hit_assignment, input_coords):
            # Count number of hits per track
            # extract hits for tracks from valid mask then extract the coords
            # find max num hits and pad up to this
            # for those hits so that final input is [batch, valid_query, 3*max_num_hits]
            num_hits_per_track = track_hit_assignment.sum(dim=-1)  # [batch, num_queries]
            # Initialize output tensor
            batch_size, num_queries, _ = track_hit_assignment.shape
            track_hit_coords = torch.zeros(batch_size, num_queries, self.max_hits * input_coords.shape[-1], device=input_coords.device)
            
              # For each track that has hits assigned, gather and flatten its hit coordinates
            for b in range(batch_size):
                for q in range(num_queries):
                    if num_hits_per_track[b, q] > 0:
                        # Get indices of assigned hits
                        hit_indices = torch.where(track_hit_assignment[b, q])[0]
                        
                        # If we have more hits than max_hits, randomly select max_hits of them
                        if len(hit_indices) > self.max_hits:
                            perm = torch.randperm(len(hit_indices), device=hit_indices.device)
                            hit_indices = hit_indices[perm[:self.max_hits]]
                        
                        # Gather coordinates for selected hits
                        track_hits = input_coords[b, hit_indices]  # [num_hits, 3]
                        
                        # Handle padding - only take up to max_hits coordinates
                        num_hits_to_use = min(len(hit_indices), self.max_hits)
                        flattened_coords = track_hits[:num_hits_to_use].flatten()  # [num_hits_to_use * 3]
                        
                        # Store coordinates - the rest will remain as zeros for padding
                        track_hit_coords[b, q, :num_hits_to_use * 3] = flattened_coords

            return track_hit_coords
        
        if isinstance(track_hit_assignment, torch.Tensor):
            print("is tensor...")
        else:
            print(track_hit_assignment.keys())
            track_hit_assignment = track_hit_assignment['hit']
        
        # Get hit coordinates
        track_hit_coords = get_hit_coords(track_hit_assignment, input_coords)

        embed = torch.cat([input_embed, track_hit_coords], dim=-1)

        # Now track_hit_coords has shape [batch, num_queries, max_hits*3]
        # with coordinates of assigned hits, padded with zeros

        # Check input dimensions
        assert input_embed.dim() == 3, f"Expected 3D input tensor, got shape {input_embed.shape}"
        
        # Get predictions from network
        preds = self.net(embed)
        # preds = self.net(input_embed)
        
        # Ensure predictions have expected shape
        assert preds.shape[-1] == self.output_dim, \
            f"Expected {self.output_dim} output features but got {preds.shape[-1]}"
        
        # Add momentum if requested
        if self.add_momentum:
            preds = self.add_momentum_to_preds(preds)
            
        return {self.name: preds}