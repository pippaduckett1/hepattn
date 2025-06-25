import torch
import torch.nn.functional as F


def focal_loss(pred_logits, targets, balance=True, gamma=2.0, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), reduction="none")
    p_t = pred * targets + (1 - pred) * (1 - targets)
    losses = ce_loss * ((1 - p_t) ** gamma)

    if balance:
        alpha = 1 - targets.float().mean()
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        losses = alpha_t * losses

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()

def object_ce_loss(pred_logits, true, mask=None, weight=None):  # noqa: ARG001
    # TODO: Add support for mask?
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight)
    return losses.mean()


def object_ce_costs(pred_logits, true):
    losses = F.binary_cross_entropy_with_logits(
        pred_logits.unsqueeze(2).expand(-1, -1, true.shape[1]), true.unsqueeze(1).expand(-1, pred_logits.shape[1], -1), reduction="none"
    )
    return losses


def mask_dice_loss(pred_logits, true, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    num = 2 * (pred * true)
    den = (pred.sum(-1) + true.sum(-1)).unsqueeze(-1)
    losses = 1 - (num + 1) / (den + 1)

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_dice_costs(pred_logits, true):
    pred = pred_logits.sigmoid()
    num = 2 * torch.einsum("bnc,bmc->bnm", pred, true)
    den = pred.sum(-1).unsqueeze(2) + true.sum(-1).unsqueeze(1)
    losses = 1 - (num + 1) / (den + 1)
    return losses


def mask_focal_loss(pred_logits, true, alpha=-1.0, gamma=2.0, mask=None, weight=None):
    pred = pred_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, true, reduction="none")
    p_t = pred * true + (1 - pred) * (1 - true)
    losses = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * true + (1 - alpha) * (1 - true)
        losses = alpha_t * losses

    if weight is not None:
        losses *= weight

    if mask is not None:
        losses = losses[mask]

    return losses.mean()


def mask_focal_costs(pred_logits, true, alpha=-1.0, gamma=2.0):
    pred = pred_logits.sigmoid()
    focal_pos = ((1 - pred) ** gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred), reduction="none")
    focal_neg = (pred**gamma) * F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred), reduction="none")
    if alpha >= 0:
        focal_pos *= alpha
        focal_neg *= 1 - alpha
    losses = torch.einsum("bnc,bmc->bnm", focal_pos, true) + torch.einsum("bnc,bmc->bnm", focal_neg, (1 - true))
    return losses


def mask_ce_loss(pred_logits, true, mask=None, weight=None):
    losses = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight, reduction="none")

    if mask is not None:
        losses = losses[mask]

    return losses.mean()

def mask_ce_costs(pred_logits, true):
    pos = F.binary_cross_entropy_with_logits(pred_logits, torch.ones_like(pred_logits), reduction="none")
    neg = F.binary_cross_entropy_with_logits(pred_logits, torch.zeros_like(pred_logits), reduction="none")
    losses = torch.einsum("bnc,bmc->bnm", pos, true) + torch.einsum("bnc,bmc->bnm", neg, (1 - true))
    return losses



def fast_track_hit_sampling(
    track_hit_logits, 
    track_hit_targets, 
    weights,
    num_points_per_track=20,
    importance_sample_ratio=0.75,
):
    """
    FAST vectorized sampling for track-hit assignments.
    Simplified version that handles all tracks uniformly.
    
    Args:
        track_hit_logits (Tensor): Predicted logits of shape (batch, num_tracks, num_hits)
        track_hit_targets (Tensor): Target assignments of shape (batch, num_tracks, num_hits)
        num_points_per_track (int): Number of hits to sample per track
        importance_sample_ratio (float): Ratio of uncertain hits to sample
    
    Returns:
        sampled_logits (Tensor): Sampled predictions of shape (batch, num_tracks, num_points_per_track)
        sampled_targets (Tensor): Sampled targets of shape (batch, num_tracks, num_points_per_track)
    """
    batch_size, num_tracks, num_hits = track_hit_logits.shape
    
    # Calculate uncertainties
    uncertainties = -(torch.abs(track_hit_logits))
    
    # Determine number of uncertain vs random hits
    num_uncertain_hits = int(importance_sample_ratio * num_points_per_track)
    num_random_hits = num_points_per_track - num_uncertain_hits
    
    # Initialize output tensors
    sampled_logits = torch.zeros(batch_size, num_tracks, num_points_per_track, device=track_hit_logits.device)
    sampled_targets = torch.zeros(batch_size, num_tracks, num_points_per_track, device=track_hit_targets.device)
    sampled_weights = torch.zeros(batch_size, num_tracks, num_points_per_track, device=track_hit_targets.device)
    
    # Vectorized sampling of uncertain hits (highest uncertainty)
    if num_uncertain_hits > 0:
        # Get top-k uncertain hits per track (vectorized)
        uncertain_values, uncertain_indices = torch.topk(
            uncertainties, 
            k=min(num_uncertain_hits, num_hits), 
            dim=-1
        )
        
        # Gather the corresponding logits and targets
        batch_indices = torch.arange(batch_size, device=track_hit_logits.device).unsqueeze(1).unsqueeze(2)
        track_indices = torch.arange(num_tracks, device=track_hit_logits.device).unsqueeze(0).unsqueeze(2)
        
        # Use advanced indexing for efficient gathering
        uncertain_logits = track_hit_logits[batch_indices, track_indices, uncertain_indices]
        uncertain_targets = track_hit_targets[batch_indices, track_indices, uncertain_indices]
        
        # Store in output tensors
        sampled_logits[:, :, :num_uncertain_hits] = uncertain_logits
        sampled_targets[:, :, :num_uncertain_hits] = uncertain_targets

        if weights is not None:
            uncertain_weights = weights[batch_indices, track_indices, uncertain_indices]
            sampled_weights[:, :, :num_uncertain_hits] = uncertain_weights

    
    # Vectorized random sampling
    if num_random_hits > 0:
        # Generate random indices for all tracks at once
        random_indices = torch.randint(
            0, num_hits, 
            (batch_size, num_tracks, num_random_hits), 
            device=track_hit_logits.device
        )
        
        # Gather random hits
        batch_indices = torch.arange(batch_size, device=track_hit_logits.device).unsqueeze(1).unsqueeze(2)
        track_indices = torch.arange(num_tracks, device=track_hit_logits.device).unsqueeze(0).unsqueeze(2)
        
        random_logits = track_hit_logits[batch_indices, track_indices, random_indices]
        random_targets = track_hit_targets[batch_indices, track_indices, random_indices]
        
        # Store in output tensors
        start_idx = num_uncertain_hits
        end_idx = start_idx + num_random_hits
        sampled_logits[:, :, start_idx:end_idx] = random_logits
        sampled_targets[:, :, start_idx:end_idx] = random_targets

        if weights is not None:
            random_weights = weights[batch_indices, track_indices, random_indices]
            sampled_weights[:, :, start_idx:end_idx] = random_weights
    
    return sampled_logits, sampled_targets, sampled_weights



def point_sampled_mask_ce_loss(
    track_hit_logits, 
    track_hit_targets, 
    num_points_per_track=20,
    importance_sample_ratio=0.75,
    weight=None
):
    """
    FAST cross-entropy loss for track-hit assignments using vectorized sampling.
    Actually reduces memory and compute time.
    
    Args:
        track_hit_logits (Tensor): Predicted logits of shape (batch, num_tracks, num_hits)
        track_hit_targets (Tensor): Target assignments of shape (batch, num_tracks, num_hits)
        num_points_per_track (int): Number of hits to sample per track
        importance_sample_ratio (float): Ratio of uncertain hits to sample
        weight (Tensor, optional): Weight for each sample
    
    Returns:
        loss (Tensor): Computed cross-entropy loss
    """
    # Fast sampling
    sampled_logits, sampled_targets, sampled_weights = fast_track_hit_sampling(
        track_hit_logits, 
        track_hit_targets, 
        weight,
        num_points_per_track,
        importance_sample_ratio
    )
    
    # Compute loss on sampled data (much smaller tensors)
    losses = F.binary_cross_entropy_with_logits(sampled_logits, sampled_targets, weight=sampled_weights, reduction="none")
    
    return losses.mean()


def point_sampled_mask_focal_loss(
    track_hit_logits, 
    track_hit_targets, 
    alpha=-1.0, 
    gamma=2.0,
    num_points_per_track=20,
    importance_sample_ratio=0.75,
    weight=None,
    mask=None,
):
    """
    FAST focal loss for track-hit assignments using vectorized sampling.
    
    Args:
        track_hit_logits (Tensor): Predicted logits of shape (batch, num_tracks, num_hits)
        track_hit_targets (Tensor): Target assignments of shape (batch, num_tracks, num_hits)
        alpha (float): Alpha parameter for focal loss
        gamma (float): Gamma parameter for focal loss
        num_points_per_track (int): Number of hits to sample per track
        importance_sample_ratio (float): Ratio of uncertain hits to sample
        weight (Tensor, optional): Weight for each sample
    
    Returns:
        loss (Tensor): Computed focal loss
    """
    # Fast sampling
    sampled_logits, sampled_targets, sampled_weights = fast_track_hit_sampling(
        track_hit_logits, 
        track_hit_targets, 
        weight,
        num_points_per_track,
        importance_sample_ratio
    )
    
    # Compute focal loss on sampled data
    pred = sampled_logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(sampled_logits, sampled_targets, reduction="none")
    p_t = pred * sampled_targets + (1 - pred) * (1 - sampled_targets)
    losses = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * sampled_targets + (1 - alpha) * (1 - sampled_targets)
        losses = alpha_t * losses
    
    if weight is not None:
        losses *= sampled_weights
    
    return losses.mean()


def point_sampled_mask_dice_loss(
    track_hit_logits, 
    track_hit_targets, 
    num_points_per_track=20,
    importance_sample_ratio=0.75,
    mask=None,
    weight=None
):
    """
    FAST dice loss for track-hit assignments using vectorized sampling.
    
    Args:
        track_hit_logits (Tensor): Predicted logits of shape (batch, num_tracks, num_hits)
        track_hit_targets (Tensor): Target assignments of shape (batch, num_tracks, num_hits)
        num_points_per_track (int): Number of hits to sample per track
        importance_sample_ratio (float): Ratio of uncertain hits to sample
        weight (Tensor, optional): Weight for each sample
    
    Returns:
        loss (Tensor): Computed dice loss
    """
    # Fast sampling
    sampled_logits, sampled_targets, sampled_weights = fast_track_hit_sampling(
        track_hit_logits, 
        track_hit_targets, 
        weight,
        num_points_per_track,
        importance_sample_ratio,
    )
    
    # Compute dice loss on sampled data
    pred = sampled_logits.sigmoid()
    num = 2 * (pred * sampled_targets)
    den = (pred.sum(-1) + sampled_targets.sum(-1)).unsqueeze(-1)
    losses = 1 - (num + 1) / (den + 1)
    
    if weight is not None:
        losses *= sampled_weights
    
    return losses.mean()


cost_fns = {
    "object_ce": object_ce_costs,
    "mask_ce": mask_ce_costs,
    "mask_dice": mask_dice_costs,
    "mask_focal": mask_focal_costs,
}

loss_fns = {
    "object_ce": object_ce_loss, 
    "mask_ce": mask_ce_loss, 
    "mask_dice": mask_dice_loss, 
    "mask_focal": mask_focal_loss,
    # Point-sampled versions
    "point_sampled_object_ce": point_sampled_mask_ce_loss,
    "point_sampled_mask_focal": point_sampled_mask_focal_loss,
    "point_sampled_mask_dice": point_sampled_mask_dice_loss,
}