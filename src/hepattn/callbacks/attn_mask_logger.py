import uuid
from contextlib import suppress
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from matplotlib.colors import ListedColormap

from hepattn.utils.local_ca import auto_local_ca_mask


class AttnMaskLogger(Callback):
    def __init__(
        self,
        log_train: bool = True,
        log_val: bool = True,
        log_stats: bool = False,
        log_every_n_batches: int = 1000,
        lca_window_sizes: list[int] | None = None,
        log_diagonal_metrics: bool = True,
        log_invalid_query_phi: bool = False,
        invalid_query_phi_bins: int = 64,
        log_query_phi_metrics: bool = False,
        log_diagonal_regression: bool = False,
        log_phi_distance_mask: bool = False,
        phi_distance_threshold: float = 0.15,
        log_query_vs_truth_phi: bool = False,
        log_predicted_masks: bool = False,
        predicted_mask_threshold: float = 0.5,
        log_all_layers: bool = False,
        log_query_mask_phi_distribution: bool = False,
        log_attn_mask_with_query_overlay: bool = False,
        # Selective mask logging - set to False to reduce image count
        log_lca_mask: bool = True,
        log_task_mask: bool = True,
        log_kv_mask: bool = True,
        log_diagnostic_masks: bool = True,
        log_attn_weights: bool = True,
    ):
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val
        self.log_stats = log_stats
        self.log_every_n_batches = log_every_n_batches
        self.lca_window_sizes = lca_window_sizes if lca_window_sizes is not None else [32, 64, 128, 512, 1024, 2048]
        self.log_diagonal_metrics = log_diagonal_metrics
        self.log_invalid_query_phi = log_invalid_query_phi
        self.invalid_query_phi_bins = invalid_query_phi_bins
        self.log_query_phi_metrics = log_query_phi_metrics
        self.log_diagonal_regression = log_diagonal_regression
        self.log_phi_distance_mask = log_phi_distance_mask
        self.phi_distance_threshold = phi_distance_threshold
        self.log_query_vs_truth_phi = log_query_vs_truth_phi
        self.log_predicted_masks = log_predicted_masks
        self.predicted_mask_threshold = predicted_mask_threshold
        self.log_all_layers = log_all_layers
        self.log_query_mask_phi_distribution = log_query_mask_phi_distribution
        self.log_attn_mask_with_query_overlay = log_attn_mask_with_query_overlay
        self.log_lca_mask = log_lca_mask
        self.log_task_mask = log_task_mask
        self.log_kv_mask = log_kv_mask
        self.log_diagnostic_masks = log_diagnostic_masks
        self.log_attn_weights = log_attn_weights

    def _log_attention_mask(
        self,
        pl_module,
        mask,
        step,
        layer,
        prefix="local_ca_mask",
        query_phi=None,
        key_phi=None,
        query_mask=None,
        invalid_mask=None,
    ):
        """Helper method to create and log attention mask figures."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        cmap = ListedColormap(["#002b7f", "#ffff33"])  # blue for 0, yellow for 1
        im = ax.imshow(mask.numpy().astype(int), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

        # Determine if phi is ascending or descending with index
        query_phi_ascending = True
        key_phi_ascending = True
        if query_phi is not None and query_phi.numel() > 1:
            query_phi_ascending = query_phi[-1].item() > query_phi[0].item()
        if key_phi is not None and key_phi.numel() > 1:
            key_phi_ascending = key_phi[-1].item() > key_phi[0].item()

        # Flip y-axis so lowest phi is at the bottom (only if phi is ascending with index)
        if query_phi_ascending:
            ax.invert_yaxis()

        # Add colorbar with clear labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label("Attention Mask", rotation=270, labelpad=15)
        cbar.ax.set_yticklabels(["Masked (0)", "Used in Attention (1)"])

        # Add title with step and layer info
        ax.set_title(f"Attention Mask - Step {step}, Layer {layer}")

        # Add arrows to axis labels to indicate phi direction
        x_arrow = "→" if key_phi_ascending else "←"
        y_arrow = "↑" if query_phi_ascending else "↓"
        ax.set_xlabel(f"Hits ({x_arrow} increasing φ)")
        ax.set_ylabel(f"Queries ({y_arrow} increasing φ)")

        if key_phi is not None and key_phi.numel() == mask.shape[1]:
            tick_idx = np.linspace(0, mask.shape[1] - 1, num=min(6, mask.shape[1]), dtype=int)
            ax.set_xticks(tick_idx)
            xticklabels = [f"{key_phi[idx].item():.2f}" for idx in tick_idx]
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        if query_phi is not None and query_phi.numel() == mask.shape[0]:
            tick_idy = np.linspace(0, mask.shape[0] - 1, num=min(6, mask.shape[0]), dtype=int)
            ax.set_yticks(tick_idy)
            yticklabels = [f"{query_phi[idx].item():.2f}" for idx in tick_idy]
            ax.set_yticklabels(yticklabels)
        overlay_mask = None
        if invalid_mask is not None:
            overlay_mask = invalid_mask
        elif query_mask is not None:
            overlay_mask = ~query_mask
        if overlay_mask is not None:
            if isinstance(overlay_mask, torch.Tensor):
                overlay_np = overlay_mask.detach().cpu().numpy().astype(bool)
            else:
                overlay_np = np.array(overlay_mask, dtype=bool)
            if overlay_np.ndim > 0 and overlay_np.shape[0] == mask.shape[0]:
                invalid_idx = np.where(overlay_np)[0]
                for idx in invalid_idx:
                    # Draw a visual overlay to mark invalid queries - this does NOT mask the data
                    # Using magenta/purple color to distinguish from the attention mask colors
                    ax.axhspan(idx - 0.5, idx + 0.5, color="magenta", alpha=0.3, linewidth=0)
        # Log directly to Comet
        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(figure_name=f"{prefix}_step{step}_layer{layer}", figure=fig, step=step)
        plt.close(fig)

    def _log_attention_weights(
        self,
        pl_module,
        weights,
        step,
        layer,
        prefix="attn_weights",
        query_phi=None,
        key_phi=None,
    ):
        """Log attention weights as a heatmap."""
        fig, ax = plt.subplots(constrained_layout=True, dpi=300)

        # Use a continuous colormap for weights
        im = ax.imshow(weights.numpy(), aspect="auto", cmap="viridis", vmin=0, interpolation="nearest")

        # Determine phi ordering for axis inversion
        query_phi_ascending = True
        key_phi_ascending = True
        if query_phi is not None and query_phi.numel() > 1:
            query_phi_ascending = query_phi[-1].item() > query_phi[0].item()
        if key_phi is not None and key_phi.numel() > 1:
            key_phi_ascending = key_phi[-1].item() > key_phi[0].item()

        if query_phi_ascending:
            ax.invert_yaxis()

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Weight", rotation=270, labelpad=15)

        ax.set_title(f"Attention Weights - Step {step}, Layer {layer}")

        x_arrow = "→" if key_phi_ascending else "←"
        y_arrow = "↑" if query_phi_ascending else "↓"
        ax.set_xlabel(f"Keys ({x_arrow} increasing φ)")
        ax.set_ylabel(f"Queries ({y_arrow} increasing φ)")

        if key_phi is not None and key_phi.numel() == weights.shape[1]:
            tick_idx = np.linspace(0, weights.shape[1] - 1, num=min(6, weights.shape[1]), dtype=int)
            ax.set_xticks(tick_idx)
            xticklabels = [f"{key_phi[idx].item():.2f}" for idx in tick_idx]
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        if query_phi is not None and query_phi.numel() == weights.shape[0]:
            tick_idy = np.linspace(0, weights.shape[0] - 1, num=min(6, weights.shape[0]), dtype=int)
            ax.set_yticks(tick_idy)
            yticklabels = [f"{query_phi[idx].item():.2f}" for idx in tick_idy]
            ax.set_yticklabels(yticklabels)

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(figure_name=f"{prefix}_step{step}_layer{layer}", figure=fig, step=step)
        plt.close(fig)

    def _log_attention_stats(self, pl_module, mask, step, layer, prefix="val"):
        """Log basic attention mask statistics."""
        try:
            # mask: shape [num_queries, num_constituents], dtype=bool or int
            hits_per_query = mask.sum(dim=1).cpu().numpy()  # shape: [num_queries]
            avg_hits_per_query = hits_per_query.mean()

            logger = getattr(pl_module, "logger", None)
            if logger is not None and hasattr(logger, "experiment"):
                logger.experiment.log_metrics(
                    {
                        f"{prefix}/attn_mask_avg_hits_per_query_layer{layer}": float(avg_hits_per_query),
                        f"{prefix}/attn_mask_max_hits_per_query_layer{layer}": float(np.max(hits_per_query)),
                        f"{prefix}/attn_mask_min_hits_per_query_layer{layer}": float(np.min(hits_per_query)),
                        f"{prefix}/attn_mask_std_hits_per_query_layer{layer}": float(np.std(hits_per_query)),
                    },
                    step=step,
                )
            else:
                print(f"[AttnMaskLogger] Step {step} Layer {layer} - Avg hits per query: {avg_hits_per_query}")
        except (ValueError, AttributeError, TypeError) as e:
            print(f"[AttnMaskLogger] Error logging attention stats: {e}")

    def _log_invalid_query_phi_distribution(self, pl_module, query_phi, invalid_mask, step, layer, prefix):
        query_phi = query_phi.detach().cpu()
        invalid_mask = invalid_mask.detach().cpu()
        invalid_phi = query_phi[invalid_mask]

        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        if invalid_phi.numel() == 0:
            ax.text(0.5, 0.5, "No invalid queries", ha="center", va="center")
            ax.set_xlabel("Query φ")
            ax.set_ylabel("Count")
        else:
            ax.hist(
                invalid_phi.float().numpy(),
                bins=self.invalid_query_phi_bins,
                color="#1f77b4",
                edgecolor="black",
            )
            ax.set_xlabel("Query φ")
            ax.set_ylabel("Invalid query count")
        ax.set_title(f"Invalid Query φ - Step {step}, Layer {layer}")

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(
                figure_name=f"{prefix}_invalid_query_phi_step{step}_layer{layer}",
                figure=fig,
                step=step,
            )
        plt.close(fig)

    def _log_query_vs_truth_phi(self, pl_module, query_phi, invalid_mask, diagnostics, step, layer, prefix):
        query_phi = query_phi.detach().cpu()
        invalid_mask_cpu = invalid_mask.detach().cpu()
        valid_mask = ~invalid_mask_cpu
        valid_queries = query_phi[valid_mask]
        invalid_queries = query_phi[invalid_mask_cpu]

        hit_phi = diagnostics.get("hit_phi")
        hit_valid = diagnostics.get("hit_valid")
        particle_phi = diagnostics.get("particle_phi")
        particle_valid = diagnostics.get("particle_valid")

        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        if valid_queries.numel() > 0:
            ax.hist(valid_queries.float().numpy(), bins=64, alpha=0.5, label="Valid queries")

        if invalid_queries.numel() > 0:
            ax.hist(invalid_queries.float().numpy(), bins=64, alpha=0.5, label="Invalid queries", color="#ff7f0e")

        if hit_phi is not None and hit_valid is not None:
            hp = hit_phi[0].detach().cpu().float()
            hv = hit_valid[0].detach().cpu().bool()
            hp = hp[hv].numpy()
            if hp.size > 0:
                ax.hist(hp, bins=64, alpha=0.4, label="Hits")

        if particle_phi is not None and particle_valid is not None:
            pp = particle_phi[0].detach().cpu().float()
            pv = particle_valid[0].detach().cpu().bool()
            pp = pp[pv].numpy()
            if pp.size > 0:
                ax.hist(pp, bins=64, alpha=0.4, label="Particles")

        ax.set_xlabel("φ")
        ax.set_ylabel("Count")
        ax.set_title(f"φ comparison - Step {step}, Layer {layer}")
        ax.legend()

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(
                figure_name=f"{prefix}_phi_overlay_step{step}_layer{layer}",
                figure=fig,
                step=step,
            )
        plt.close(fig)

    def _log_query_mask_phi_distribution(self, pl_module, query_phi, query_mask, step, layer, prefix, diagnostics=None):
        query_phi = query_phi.detach().cpu()
        query_mask = query_mask.detach().cpu().bool()
        invalid_mask = ~query_mask

        fig, ax = plt.subplots(constrained_layout=True, dpi=300)
        bins = min(64, max(10, query_phi.numel()))

        if query_phi.numel() == 0:
            ax.text(0.5, 0.5, "No queries", ha="center", va="center")
        else:
            valid_phi = query_phi[query_mask]
            invalid_phi = query_phi[invalid_mask]
            if valid_phi.numel() > 0:
                ax.hist(
                    valid_phi.float().numpy(),
                    bins=bins,
                    alpha=0.5,
                    label="Pred valid",
                    color="#1f77b4",
                    edgecolor="black",
                )
            if invalid_phi.numel() > 0:
                ax.hist(
                    invalid_phi.float().numpy(),
                    bins=bins,
                    alpha=0.5,
                    label="Pred invalid",
                    color="#ff7f0e",
                    edgecolor="black",
                )
            if diagnostics is not None:
                hit_phi = diagnostics.get("hit_phi")
                hit_valid = diagnostics.get("hit_valid")
                if hit_phi is not None and hit_valid is not None:
                    hp = hit_phi[0].detach().cpu().float()
                    hv = hit_valid[0].detach().cpu().bool()
                    hp = hp[hv]
                    if hp.numel() > 0:
                        ax.hist(
                            hp.numpy(),
                            bins=bins,
                            alpha=0.4,
                            label="Hit φ",
                            color="#2ca02c",
                            edgecolor="black",
                        )

        ax.set_xlabel("Query φ")
        ax.set_ylabel("Count")
        ax.set_title(f"Query mask φ dist - Step {step}, Layer {layer}")
        ax.legend()

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_figure(
                figure_name=f"{prefix}_query_mask_phi_step{step}_layer{layer}",
                figure=fig,
                step=step,
            )
        plt.close(fig)

    def _log_query_phi_metrics(self, pl_module, query_phi, key_phi, step, layer, prefix):
        query_phi = query_phi.detach().cpu()
        metrics = {
            f"{prefix}/query_phi_min_layer{layer}": float(query_phi.min().item()),
            f"{prefix}/query_phi_max_layer{layer}": float(query_phi.max().item()),
            f"{prefix}/query_phi_span_layer{layer}": float((query_phi.max() - query_phi.min()).item()),
        }

        if key_phi is not None:
            key_phi_tensor = key_phi.detach().cpu()
            key_span = (key_phi_tensor.max() - key_phi_tensor.min()).item()
            metrics.update({
                f"{prefix}/key_phi_min_layer{layer}": float(key_phi_tensor.min().item()),
                f"{prefix}/key_phi_max_layer{layer}": float(key_phi_tensor.max().item()),
                f"{prefix}/key_phi_span_layer{layer}": float(key_span),
                f"{prefix}/phi_span_ratio_layer{layer}": float(key_span / (2 * np.pi)),
            })

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_metrics(metrics, step=step)

    def _log_diagonal_regression_metrics(self, pl_module, mask, step, layer, prefix):
        mask = mask.bool()
        num_queries, num_hits = mask.shape
        hit_idx = torch.arange(num_hits, device=mask.device, dtype=torch.float32)
        query_idx = torch.arange(num_queries, device=mask.device, dtype=torch.float32)

        hits_per_query = mask.sum(dim=1).float()
        valid = hits_per_query > 0
        if not valid.any():
            return

        mean_hits = (mask.float() * hit_idx).sum(dim=1)
        mean_hits[valid] /= hits_per_query[valid]

        q_valid = query_idx[valid]
        k_valid = mean_hits[valid]

        q_centered = q_valid - q_valid.mean()
        k_centered = k_valid - k_valid.mean()
        var = (q_centered**2).mean()
        if var == 0:
            return
        cov = (q_centered * k_centered).mean()
        slope = cov / var
        intercept = k_valid.mean() - slope * q_valid.mean()
        residuals = k_valid - (slope * q_valid + intercept)
        rmse = torch.sqrt((residuals**2).mean())
        expected_slope = num_hits / max(num_queries, 1)

        metrics = {
            f"{prefix}/diag_reg_slope_layer{layer}": float(slope.item()),
            f"{prefix}/diag_reg_intercept_layer{layer}": float(intercept.item()),
            f"{prefix}/diag_reg_rmse_layer{layer}": float(rmse.item()),
            f"{prefix}/diag_expected_slope_layer{layer}": float(expected_slope),
        }

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_metrics(metrics, step=step)

    def _log_phi_distance_mask(
        self,
        pl_module,
        query_phi,
        key_phi,
        attn_mask,
        step,
        layer,
        prefix,
        query_mask=None,
        invalid_mask=None,
    ):
        query_phi = query_phi.to(device=key_phi.device if isinstance(key_phi, torch.Tensor) else query_phi.device)
        query_phi = query_phi.unsqueeze(-1)
        key_phi = key_phi.unsqueeze(-2)
        diff = query_phi - key_phi
        diff = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()
        phi_mask = diff <= self.phi_distance_threshold

        phi_mask_cpu = phi_mask.detach().cpu()
        self._log_attention_mask(
            pl_module,
            phi_mask_cpu.int(),
            step,
            layer,
            f"phi_distance_mask_{prefix}",
            query_phi=query_phi.squeeze(-1).detach().cpu(),
            key_phi=key_phi.squeeze(-2).detach().cpu(),
            query_mask=query_mask,
            invalid_mask=invalid_mask,
        )

        attn_bool = attn_mask.bool()
        phi_bool = phi_mask_cpu.bool()
        intersection = (attn_bool & phi_bool).sum().item()
        phi_total = phi_bool.sum().item()
        attn_total = attn_bool.sum().item()

        metrics = {
            f"{prefix}/phi_mask_intersection_layer{layer}": intersection,
            f"{prefix}/phi_mask_efficiency_layer{layer}": float(intersection / attn_total) if attn_total > 0 else 0.0,
            f"{prefix}/phi_mask_purity_layer{layer}": float(intersection / phi_total) if phi_total > 0 else 0.0,
        }

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_metrics(metrics, step=step)

    def _calculate_multi_lca_comparison_metrics(self, ma_mask):
        """Calculate LCA comparison metrics for multiple window sizes using dummy embeddings."""
        all_metrics = {}

        # Extract dimensions from the attention mask
        num_queries, num_hits = ma_mask.shape
        device = ma_mask.device

        # Create dummy embeddings with correct shapes
        # We use any tensor with the right shape[1] - the values don't matter for LCA mask generation
        dummy_q_embed = torch.zeros(1, num_queries, device=device)  # batch_size=1, num_queries
        dummy_kv_embed = torch.zeros(1, num_hits, device=device)  # batch_size=1, num_hits

        for window_size in self.lca_window_sizes:
            # Generate LCA mask for this window size
            lca_mask = auto_local_ca_mask(dummy_q_embed, dummy_kv_embed, window_size, wrap=True)
            lca_mask = lca_mask.squeeze(0)  # Remove batch dimension

            # Calculate efficiency: fraction of MA mask positions that are in LCA mask
            # Efficiency = (MA ∩ LCA) / MA
            ma_positions = ma_mask.sum()
            intersection = (ma_mask & lca_mask).sum()
            efficiency = float(intersection / ma_positions) if ma_positions > 0 else 0.0

            # Calculate purity: fraction of LCA mask positions that are in MA mask
            # Purity = (MA ∩ LCA) / LCA
            lca_positions = lca_mask.sum()
            purity = float(intersection / lca_positions) if lca_positions > 0 else 0.0

            # Store metrics with window size suffix
            window_suffix = f"_w{window_size}"
            all_metrics.update({
                f"attn_mask_lca_efficiency{window_suffix}": efficiency,
                f"attn_mask_lca_purity{window_suffix}": purity,
                f"attn_mask_intersection{window_suffix}": float(intersection),
            })

        return all_metrics

    def _calculate_distance_from_diagonal(self, mask):
        """Calculate how close the attention pattern is to a diagonal band."""
        # Using a vectorized PyTorch implementation for performance.
        num_queries, num_hits = mask.shape
        if num_queries == 0:
            return 0.0

        stride = num_hits / num_queries

        # Get indices of attended hits
        query_indices, hit_indices = torch.where(mask.bool())
        if query_indices.numel() == 0:
            return 0.0

        # Calculate expected diagonal hit index for each attended hit
        strided_query_indices = torch.round(query_indices.float() * stride)

        # Calculate distances
        distances = torch.abs(hit_indices.float() - strided_query_indices)

        # To calculate mean of means, we sum distances per query and divide by hits per query
        sum_distances_per_query = torch.zeros(num_queries, device=mask.device, dtype=torch.float)
        sum_distances_per_query.scatter_add_(0, query_indices, distances)

        hits_per_query = mask.sum(dim=1).float()

        # Queries with hits
        has_hits_mask = hits_per_query > 0
        if not has_hits_mask.any():
            return 0.0

        # Calculate average distance per query, avoiding division by zero
        avg_distance_per_query = torch.zeros_like(hits_per_query)
        avg_distance_per_query[has_hits_mask] = sum_distances_per_query[has_hits_mask] / hits_per_query[has_hits_mask]

        # Average of these averages
        avg_diagonal_distance = avg_distance_per_query[has_hits_mask].mean()

        # Normalize
        max_distance = num_hits / 2
        if max_distance == 0:
            return 0.0
        return float(avg_diagonal_distance / max_distance)

    def _calculate_diagonal_band_width(self, mask):
        """Calculate the average width of the diagonal band in the attention mask."""
        # Using a vectorized PyTorch implementation for performance.
        num_queries, num_hits = mask.shape
        if num_queries == 0:
            return 0.0

        has_hits = mask.any(dim=1)
        if not has_hits.any():
            return 0.0

        col_indices = torch.arange(num_hits, device=mask.device)
        masked_indices = col_indices.expand_as(mask)

        # For min, set non-attended to a large value
        first_hits = torch.where(mask.bool(), masked_indices, num_hits)
        min_attended_indices = torch.min(first_hits, dim=1).values[has_hits]

        # For max, set non-attended to a small value
        last_hits = torch.where(mask.bool(), masked_indices, -1)
        max_attended_indices = torch.max(last_hits, dim=1).values[has_hits]

        # Calculate widths
        widths = max_attended_indices - min_attended_indices + 1

        return float(widths.float().mean())

    def _calculate_attention_consistency(self, mask):
        """Calculate how consistent the attention pattern is across queries."""
        # Convert to numpy for easier processing
        mask_np = mask.cpu().numpy().astype(bool)

        # Calculate the number of hits each query attends to
        hits_per_query = mask_np.sum(axis=1)

        # Calculate consistency as 1 - coefficient of variation
        if len(hits_per_query) > 1:
            mean_hits = np.mean(hits_per_query)
            std_hits = np.std(hits_per_query)
            if mean_hits > 0:
                cv = std_hits / mean_hits
                consistency = max(0.0, 1.0 - cv)
                return float(consistency)

        return 1.0  # If all queries attend to the same number of hits

    def _log_diagonal_metrics(self, pl_module, ma_mask, step, layer, prefix="val"):
        """Log metrics comparing MA mask to LCA mask to measure diagonal consistency."""
        # Calculate basic metrics based on MA mask structure
        diagonal_distance = self._calculate_distance_from_diagonal(ma_mask)
        ma_positions = ma_mask.sum()
        diagonal_band_width = self._calculate_diagonal_band_width(ma_mask)
        consistency = self._calculate_attention_consistency(ma_mask)

        metrics = {
            f"{prefix}/attn_mask_distance_from_diagonal_layer{layer}": diagonal_distance,
            f"{prefix}/attn_mask_diagonal_band_width_layer{layer}": diagonal_band_width,
            f"{prefix}/attn_mask_consistency_layer{layer}": consistency,
            f"{prefix}/attn_mask_ma_positions_layer{layer}": float(ma_positions),
        }

        # Calculate LCA comparison metrics for all window sizes using dummy embeddings
        lca_metrics = self._calculate_multi_lca_comparison_metrics(ma_mask)
        metrics.update(lca_metrics)

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            logger.experiment.log_metrics(metrics, step=step)

    def _process_attention_masks_from_outputs(self, pl_module, outputs, step, is_validation=False):
        """Process attention masks directly from the outputs dictionary."""
        prefix_suffix = "_val" if is_validation else "train"
        diagnostics = outputs.get("diagnostics")

        # Get only entries that contain "attn_mask"
        layer_outputs = {k: v for k, v in outputs.items() if k != "loss" and "attn_mask" in v}
        if not layer_outputs:
            return

        layer_indices = sorted(int(k.split("_")[1]) for k in layer_outputs)
        if not layer_indices:
            return

        for layer_name, l_out in outputs.items():
            if layer_name != "loss" and "attn_mask" in l_out:
                layer_index = int(layer_name.split("_")[1])
                query_phi = l_out.get("query_phi")
                key_phi = l_out.get("key_phi")
                invalid_mask = l_out.get("query_invalid_mask")
                query_mask = l_out.get("query_mask")
                lca_mask = l_out.get("lca_mask")
                task_mask = l_out.get("task_attn_mask")

                if self.log_all_layers or layer_index == max(layer_indices):
                    attn_mask = l_out["attn_mask"]
                    attn_mask_im = attn_mask[0].detach().cpu().clone().int()
                    query_sample = query_phi[0].detach().cpu() if query_phi is not None else None
                    key_sample = key_phi[0].detach().cpu() if key_phi is not None else None
                    query_mask_sample = None
                    if query_mask is not None:
                        query_mask_sample = query_mask[0].detach().cpu().bool()
                    invalid_mask_sample = None
                    if invalid_mask is not None:
                        invalid_mask_sample = invalid_mask[0].detach().cpu().bool()
                    elif query_mask_sample is not None:
                        invalid_mask_sample = ~query_mask_sample
                    self._log_attention_mask(
                        pl_module,
                        attn_mask_im,
                        step,
                        layer_index,
                        f"local_ma_mask_{prefix_suffix}",
                        query_phi=query_sample,
                        key_phi=key_sample,
                        query_mask=query_mask_sample,
                        invalid_mask=invalid_mask_sample,
                    )
                    if self.log_kv_mask:
                        kv_mask = l_out.get("attn_mask_kv")
                        if kv_mask is not None:
                            kv_im = kv_mask[0].detach().cpu().clone().int()
                            self._log_attention_mask(
                                pl_module,
                                kv_im,
                                step,
                                layer_index,
                                f"local_ma_mask_kv_{prefix_suffix}",
                                query_phi=key_sample,
                                key_phi=query_sample,
                            )
                    if self.log_lca_mask and lca_mask is not None:
                        lca_im = lca_mask[0].detach().cpu().clone().int()
                        self._log_attention_mask(
                            pl_module,
                            lca_im,
                            step,
                            layer_index,
                            f"local_lca_mask_{prefix_suffix}",
                            query_phi=query_sample,
                            key_phi=key_sample,
                            query_mask=query_mask_sample,
                            invalid_mask=invalid_mask_sample,
                        )
                    if self.log_task_mask and task_mask is not None:
                        task_im = task_mask[0].detach().cpu().clone().int()
                        self._log_attention_mask(
                            pl_module,
                            task_im,
                            step,
                            layer_index,
                            f"local_task_mask_{prefix_suffix}",
                            query_phi=query_sample,
                            key_phi=key_sample,
                            query_mask=query_mask_sample,
                            invalid_mask=invalid_mask_sample,
                        )

                    # Log diagnostic task masks if present (after_ca, after_sa, after_bidi)
                    if self.log_diagnostic_masks:
                        for diag_stage in ["after_ca", "after_sa", "after_bidi"]:
                            for diag_key in l_out:
                                if diag_key.startswith(f"{diag_stage}_"):
                                    diag_mask = l_out[diag_key]
                                    if diag_mask is not None and torch.is_tensor(diag_mask):
                                        diag_im = diag_mask[0].detach().cpu().clone().int()
                                        self._log_attention_mask(
                                            pl_module,
                                            diag_im,
                                            step,
                                            layer_index,
                                            f"diag_{diag_key}_{prefix_suffix}",
                                            query_phi=query_sample,
                                            key_phi=key_sample,
                                            query_mask=query_mask_sample,
                                            invalid_mask=invalid_mask_sample,
                                        )

                    # Log attention weights if present
                    if self.log_attn_weights:
                        fwd_attn_weights = l_out.get("fwd_ca_attn_weights")
                        if fwd_attn_weights is not None:
                            self._log_attention_weights(
                                pl_module,
                                fwd_attn_weights[0].detach().cpu(),
                                step,
                                layer_index,
                                f"fwd_ca_attn_weights_{prefix_suffix}",
                                query_phi=query_sample,
                                key_phi=key_sample,
                            )

                        bidi_attn_weights = l_out.get("bidi_ca_attn_weights")
                        if bidi_attn_weights is not None:
                            # Note: for bidi CA, rows are keys, cols are queries
                            self._log_attention_weights(
                                pl_module,
                                bidi_attn_weights[0].detach().cpu(),
                                step,
                                layer_index,
                                f"bidi_ca_attn_weights_{prefix_suffix}",
                                query_phi=key_sample,  # Rows are keys
                                key_phi=query_sample,  # Cols are queries
                            )
                    if step > 10000:
                        self._log_mask_points_for_kde(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")
                    if self.log_stats:
                        self._log_attention_stats(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")

                    # Log diagonal metrics if enabled
                    if self.log_diagonal_metrics:
                        self._log_diagonal_metrics(pl_module, attn_mask_im, step, layer_index, f"local_ma_mask_{prefix_suffix}")

                    if self.log_query_phi_metrics and query_phi is not None:
                        key_sample = key_phi[0] if key_phi is not None else None
                        self._log_query_phi_metrics(
                            pl_module,
                            query_phi[0],
                            key_sample,
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                        )

                    if self.log_diagonal_regression:
                        self._log_diagonal_regression_metrics(
                            pl_module,
                            attn_mask_im,
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                        )

                    if self.log_invalid_query_phi and query_phi is not None and invalid_mask is not None:
                        self._log_invalid_query_phi_distribution(
                            pl_module,
                            query_phi[0],
                            invalid_mask[0],
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                        )

                    if self.log_phi_distance_mask and query_phi is not None and key_phi is not None:
                        self._log_phi_distance_mask(
                            pl_module,
                            query_sample,
                            key_sample,
                            attn_mask_im,
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                            query_mask=query_mask_sample,
                            invalid_mask=invalid_mask_sample,
                        )

                    if self.log_query_vs_truth_phi and query_phi is not None and invalid_mask is not None and diagnostics is not None:
                        self._log_query_vs_truth_phi(
                            pl_module,
                            query_sample,
                            invalid_mask[0],
                            diagnostics,
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                        )

                    if self.log_query_mask_phi_distribution and query_mask is not None and query_phi is not None:
                        self._log_query_mask_phi_distribution(
                            pl_module,
                            query_sample,
                            query_mask[0],
                            step,
                            layer_index,
                            f"local_ma_mask_{prefix_suffix}",
                            diagnostics=diagnostics,
                        )

                    # Log additional version with query mask overlay
                    if self.log_attn_mask_with_query_overlay:
                        # Always log the overlay plot when enabled, even if query_mask is None
                        # The overlay will only appear if query_mask exists and has False values
                        self._log_attention_mask(
                            pl_module,
                            attn_mask_im,
                            step,
                            layer_index,
                            f"local_ma_mask_with_query_overlay_{prefix_suffix}",
                            query_phi=query_sample,
                            key_phi=key_sample,
                            query_mask=query_mask_sample,
                            invalid_mask=None,  # Explicitly use query_mask, not invalid_mask
                        )

                    if "q_sa_delta" in l_out:
                        delta = l_out["q_sa_delta"][0].detach().cpu()
                        metrics = {
                            f"{prefix_suffix}/q_sa_delta_mean_layer{layer_index}": float(delta.mean().item()),
                            f"{prefix_suffix}/q_sa_delta_max_layer{layer_index}": float(delta.max().item()),
                            f"{prefix_suffix}/q_sa_delta_min_layer{layer_index}": float(delta.min().item()),
                        }
                        logger = getattr(pl_module, "logger", None)
                        if logger is not None and hasattr(logger, "experiment"):
                            logger.experiment.log_metrics(metrics, step=step)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.log_val:
            return
        step = getattr(trainer, "global_step", batch_idx)
        self._process_attention_masks_from_outputs(pl_module, outputs, step, is_validation=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train:
            return
        # only process if this batch is selected by the sampler
        if batch_idx % self.log_every_n_batches != 0:
            return
        step = getattr(trainer, "global_step", batch_idx)
        self._process_attention_masks_from_outputs(pl_module, outputs, step, is_validation=False)

    def _log_mask_points_for_kde(self, pl_module, mask, step, layer, prefix="local_ma_mask"):
        """Save a subsampled set of (query, key) coordinates where mask==1.
        These can be used later to build KDE plots without storing full masks.
        """
        # mask: [num_queries, num_hits], 0/1 or bool
        mask_bool = mask.bool()
        q_idx, k_idx = torch.where(mask_bool)

        # Nothing to save if there are no hits
        if q_idx.numel() == 0:
            return

        # Subsample for memory / disk usage
        max_points = 100_000  # tune as you like
        num_points = q_idx.numel()
        if num_points > max_points:
            perm = torch.randperm(num_points, device=mask.device)[:max_points]
            q_idx = q_idx[perm]
            k_idx = k_idx[perm]

        coords = torch.stack([q_idx, k_idx], dim=1).cpu().numpy()

        # Use small integer dtype if sequence length allows it
        coords = coords.astype("uint16") if (mask.shape[0] < 65535) and (mask.shape[1] < 65535) else coords.astype("uint32")

        # Save as compressed npz
        filename = Path(f"attn_hits_{prefix}_step{step}_layer{layer}_{uuid.uuid4().hex}.npz")
        np.savez_compressed(filename, coords=coords, num_queries=mask.shape[0], num_hits=mask.shape[1])

        logger = getattr(pl_module, "logger", None)
        if logger is not None and hasattr(logger, "experiment"):
            # For Comet: log as an asset / artifact
            logger.experiment.log_asset(file_data=filename, file_name=filename, step=step)
        # Clean up local file
        with suppress(OSError):
            filename.unlink()