"""Based on
- https://github.com/facebookresearch/MaskFormer
- https://github.com/facebookresearch/Mask2Former.
"""

from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from hepattn.flex.fast_local_ca import build_strided_sliding_window_blockmask
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped, transpose_blockmask
from hepattn.models.attention import Attention
from hepattn.models.dense import Dense
from hepattn.models.encoder import Residual
from hepattn.models.norm import get_hybrid_norm_config
from hepattn.models.posenc import pos_enc_symmetric
from hepattn.models.task import IncidenceRegressionTask, ObjectClassificationTask
from hepattn.utils.local_ca import auto_local_ca_mask
from hepattn.utils.model_utils import unmerge_inputs


class MaskFormerDecoder(nn.Module):
    def __init__(
        self,
        num_queries: int,
        decoder_layer_config: dict,
        num_decoder_layers: int,
        mask_attention: bool = True,
        use_query_masks: bool = False,
        log_query_masks_only: bool = False,
        posenc: dict[str, float] | None = None,
        local_strided_attn: bool = False,
        window_size: int = 512,
        window_wrap: bool = True,
        fast_local_ca: bool = False,
        block_size: int = 128,
        unified_decoding: bool = False,
        phi_shift: float = 0.0,
        combine_ma_lca: str | None = None,
        change_if_none_then_all: bool = False,
        use_first_layer_mask_only: bool = False,
        mask_attention_first_layer_only: bool = False,
        mask_attention_num_layers: int | None = None,
        mask_attention_start_layer: int | None = None,
        every_other_layer: bool = False,
        local_strided_skip_first_layer: bool = False,
        local_strided_start_layer: int | None = None,
        local_strided_first_layer_only: bool = False,
        local_strided_first_layer_window_size: int | None = None,
        local_strided_first_layer_decay_tau: float | None = None,
        local_strided_decay: bool = False,
        local_strided_decay_tau: float = 256.0,
        local_strided_mask_from_phi: bool = False,
        use_phi_distance_mask: bool = False,
        phi_distance_threshold: float = 0.2,
        no_if_none_then_all_first_layer_only: bool = False,
        no_if_none_then_all: bool = False,
        add_direct_pe_in_layer: bool = True,
        add_pe_pre_layer: bool = False,
        add_pe_in_attn: bool = True,
        learn_phi_shift: bool = False,
        flipped: bool = False,
        phi_shift_from_key_phi: bool = False,
        log_query_phi: bool = True,
        log_key_phi: bool = False,
        quantile_query_phi: bool = False,
        histogram_query_phi: bool = False,
        histogram_query_phi_bins: int = 64,
        lca_shift_absolute: int | None = None,
        lca_shift_fractional: float | None = None,
        lca_shift_queries: int | None = None,
        phi_aware_transpose: bool = False,
        bidirectional_ca_start_layer: int | None = None,
        bidirectional_ca_use_key_values: bool = False,
        bidirectional_ca_transpose_mode: str = "simple",
        bidirectional_ca_window_scale: float = 1.0,
        bidirectional_ca_soft_narrowing: bool = False,
        bidirectional_ca_narrowing_tau: float = 0.1,
        dynamic_queries: bool = False,
    ):
        """MaskFormer decoder that handles multiple decoder layers and task integration.

        Args:
            num_queries: The number of object-level queries (ignored when dynamic_queries=True).
            decoder_layer_config: Configuration dictionary used to initialize each MaskFormerDecoderLayer.
            num_decoder_layers: The number of decoder layers to stack.
            mask_attention: If True, attention masks will be used to control which input constituents are attended to.
            use_query_masks: If True, predicted query masks will be used to control which queries are valid.
            log_query_masks_only: If True, query masks are recorded for logging but not applied during decoding.
            posenc: Optional module for positional encoding.
            local_strided_attn: If True, uses local strided window attention.
            window_size: The size of the window for local strided window attention.
            window_wrap: If True, wraps the window for local strided window attention.
            attn_type: The attention type to use (e.g., 'torch', 'flex').
            fast_local_ca: If True, uses fast local CA.
            block_size: The size of the block for fast local CA.
            unified_decoding: If True, inputs remain merged for task processing instead of being unmerged after each layer.
        """
        super().__init__()

        self.decoder_layers = nn.ModuleList([MaskFormerDecoderLayer(depth=i, **decoder_layer_config) for i in range(num_decoder_layers)])
        self.dim = decoder_layer_config["dim"]
        self.bidirectional_ca = decoder_layer_config.get("bidirectional_ca", True)
        self.tasks: list | None = None  # Will be set by MaskFormer
        self.num_queries = num_queries
        self.mask_attention = mask_attention
        self.use_query_masks = use_query_masks
        self.log_query_masks_only = log_query_masks_only
        self.posenc = posenc
        self.local_strided_attn = local_strided_attn
        self.attn_type = decoder_layer_config.get("attn_kwargs", {}).get("attn_type", "torch")
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.unified_decoding = unified_decoding
        self.dynamic_queries = dynamic_queries
        # When dynamic_queries=True, queries are zero-initialized at forward time based on truth particle count
        # When dynamic_queries=False, queries are learnable parameters
        if not self.dynamic_queries:
            self.initial_queries = nn.Parameter(torch.randn(self.num_queries, decoder_layer_config["dim"]))
        self.fast_local_ca = fast_local_ca
        self.block_size = block_size
        self.phi_shift = phi_shift
        self.change_if_none_then_all = change_if_none_then_all
        self.use_first_layer_mask_only = use_first_layer_mask_only
        self.mask_attention_first_layer_only = mask_attention_first_layer_only
        self.mask_attention_num_layers = int(mask_attention_num_layers) if mask_attention_num_layers is not None else None
        if self.mask_attention_num_layers is not None:
            assert self.mask_attention_num_layers > 0, "mask_attention_num_layers must be positive"
        self.mask_attention_start_layer = int(mask_attention_start_layer) if mask_attention_start_layer is not None else 0
        assert self.mask_attention_start_layer >= 0, "mask_attention_start_layer must be non-negative"
        self.no_if_none_then_all = no_if_none_then_all
        self.every_other_layer = every_other_layer
        self.local_strided_skip_first_layer = local_strided_skip_first_layer
        self.local_strided_first_layer_only = local_strided_first_layer_only
        if local_strided_start_layer is not None:
            self.local_strided_start_layer = int(local_strided_start_layer)
        else:
            self.local_strided_start_layer = 1 if local_strided_skip_first_layer else 0
        assert self.local_strided_start_layer >= 0, "local_strided_start_layer must be non-negative"
        self.local_strided_first_layer_window_size = local_strided_first_layer_window_size
        self.local_strided_first_layer_decay_tau = (
            float(local_strided_first_layer_decay_tau) if local_strided_first_layer_decay_tau is not None else None
        )
        self.local_strided_decay = local_strided_decay
        self.local_strided_decay_tau = float(local_strided_decay_tau)
        self.local_strided_mask_from_phi = local_strided_mask_from_phi
        self.use_phi_distance_mask = use_phi_distance_mask
        self.phi_distance_threshold = float(phi_distance_threshold)
        self.no_if_none_then_all_first_layer_only = no_if_none_then_all_first_layer_only

        self.add_direct_pe_in_layer = add_direct_pe_in_layer
        self.add_pe_pre_layer = add_pe_pre_layer
        self.add_pe_in_attn = add_pe_in_attn

        self.flipped = flipped
        self.phi_shift_from_key_phi = phi_shift_from_key_phi
        self.log_query_phi = log_query_phi
        self.log_key_phi = log_key_phi
        self.quantile_query_phi = quantile_query_phi
        self.histogram_query_phi = histogram_query_phi
        self.histogram_query_phi_bins = int(histogram_query_phi_bins)

        if self.quantile_query_phi and self.histogram_query_phi:
            raise ValueError("quantile_query_phi and histogram_query_phi cannot both be True")
        if self.histogram_query_phi:
            assert self.histogram_query_phi_bins > 0, "histogram_query_phi_bins must be positive"

        # LCA shift parameters
        self.lca_shift_absolute = int(lca_shift_absolute) if lca_shift_absolute is not None else None
        self.lca_shift_fractional = float(lca_shift_fractional) if lca_shift_fractional is not None else None
        self.lca_shift_queries = int(lca_shift_queries) if lca_shift_queries is not None else None
        shift_count = sum(x is not None for x in [self.lca_shift_absolute, self.lca_shift_fractional, self.lca_shift_queries])
        if shift_count > 1:
            raise ValueError("Only one of lca_shift_absolute, lca_shift_fractional, or lca_shift_queries can be set")

        if learn_phi_shift:
            # learned scalar in radians fraction (same semantics as before: subtracted inside 2*pi*(.. - phi_shift))
            self.phi_shift = nn.Parameter(torch.tensor(float(phi_shift)))
        else:
            # keep it constant but device/dtype aware
            self.register_buffer("phi_shift_buffer", torch.tensor(float(phi_shift)), persistent=True)
            self.phi_shift = self.phi_shift_buffer

        if self.local_strided_attn:
            assert self.attn_type in {"torch", "flex"}, (
                f"Invalid attention type when local_strided_attn is True: {self.attn_type}, must be 'torch' or 'flex'"
            )
        if shift_count > 0 and self.attn_type == "flex":
            raise ValueError("LCA shift is not supported with flex attention")
        mask_layer_limit = None
        if self.mask_attention_first_layer_only:
            mask_layer_limit = 1
        elif self.mask_attention_num_layers is not None:
            mask_layer_limit = self.mask_attention_num_layers
        
        # Check if mask attention and local strided attention are disjoint (don't overlap in layers)
        # Case 1: mask attention is confined to early layers, LCA starts after
        ma_before_lca = mask_layer_limit is not None and self.local_strided_start_layer >= mask_layer_limit
        # Case 2: LCA is confined to early layers, mask attention starts after
        lca_end_layer = 1 if self.local_strided_first_layer_only else float('inf')
        lca_before_ma = self.mask_attention_start_layer >= lca_end_layer
        
        allow_disjoint_masks = (
            self.mask_attention and self.local_strided_attn and (ma_before_lca or lca_before_ma)
        )
        if combine_ma_lca is None:
            assert not (self.local_strided_attn and self.mask_attention) or allow_disjoint_masks, (
                "local_strided_attn and mask_attention cannot both be True without combine_ma_lca unless "
                "mask_attention is confined to layers before local strided attention begins"
            )
        if self.local_strided_decay:
            assert self.local_strided_attn, "local_strided_decay requires local_strided_attn to be True"
            assert self.attn_type == "torch", "local_strided_decay currently only supports torch attention backend"
        if self.local_strided_first_layer_window_size is not None:
            assert self.local_strided_attn, "local_strided_first_layer_window_size requires local_strided_attn to be True"
            assert self.local_strided_first_layer_window_size % 2 == 0, "local_strided_first_layer_window_size must be even"
        if self.local_strided_first_layer_decay_tau is not None:
            assert self.local_strided_first_layer_window_size is not None, (
                "local_strided_first_layer_decay_tau requires local_strided_first_layer_window_size to be set"
            )
            assert self.local_strided_decay, "local_strided_first_layer_decay_tau requires local_strided_decay to be True"

        if self.local_strided_attn and self.mask_attention and not allow_disjoint_masks:
            assert combine_ma_lca in {"OR", "AND"}, (
                "When both mask_attention and local_strided_attn are True, combine_ma_lca must be either 'OR' or 'AND'."
            )
        self.combine_ma_lca = combine_ma_lca

        # Bidirectional CA control parameters
        self.phi_aware_transpose = phi_aware_transpose
        self.bidirectional_ca_use_key_values = bidirectional_ca_use_key_values
        if bidirectional_ca_start_layer is not None:
            self.bidirectional_ca_start_layer = int(bidirectional_ca_start_layer)
        else:
            self.bidirectional_ca_start_layer = 0
        if self.phi_aware_transpose and not self.bidirectional_ca:
            raise ValueError("phi_aware_transpose requires bidirectional_ca to be True in decoder_layer_config")

        # Transpose mode for bidirectional CA attention mask
        # - "simple": standard transpose (current behavior)
        # - "phi_aware": remap based on φ positions (uses phi_aware_transpose)
        # - "direct_phi": compute K×Q mask directly from φ similarity, ignoring task mask
        # - "scaled": scale indices to account for Q≠K aspect ratio
        valid_transpose_modes = {"simple", "phi_aware", "direct_phi", "scaled"}
        if bidirectional_ca_transpose_mode not in valid_transpose_modes:
            raise ValueError(f"bidirectional_ca_transpose_mode must be one of {valid_transpose_modes}")
        self.bidirectional_ca_transpose_mode = bidirectional_ca_transpose_mode
        # For backwards compatibility, phi_aware_transpose=True overrides to "phi_aware" mode
        if self.phi_aware_transpose:
            self.bidirectional_ca_transpose_mode = "phi_aware"

        # Window scale for bidirectional CA mask (relative to forward CA)
        # Values < 1.0 create a narrower window for bidirectional attention
        # This can reduce "contamination" from neighboring queries when updating keys
        self.bidirectional_ca_window_scale = float(bidirectional_ca_window_scale)
        if self.bidirectional_ca_window_scale <= 0:
            raise ValueError("bidirectional_ca_window_scale must be positive")

        # Soft narrowing: instead of hard mask cutoff, use attention bias
        # This downweights distant queries without completely blocking them
        # Preserves gradients and allows the model to still attend to "true" owner even if distant
        self.bidirectional_ca_soft_narrowing = bidirectional_ca_soft_narrowing
        self.bidirectional_ca_narrowing_tau = float(bidirectional_ca_narrowing_tau)
        if self.bidirectional_ca_narrowing_tau <= 0:
            raise ValueError("bidirectional_ca_narrowing_tau must be positive")

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        """Forward pass through decoder layers.

        Args:
            x: Dictionary containing embeddings and masks.
            input_names: List of input names for constructing attention masks.

        Returns:
            Tuple containing updated embeddings and outputs from each decoder layer and final outputs.

        Raises:
            ValueError: If in merged input mode and multiple attention masks are provided.
        """
        batch_size = x["key_embed"].shape[0]
        num_constituents = x["key_embed"].shape[-2]

        # Generate the queries that represent objects
        if self.dynamic_queries:
            # Dynamic queries: num_queries comes from input, zero-initialized
            if "num_truth_particles" not in x["inputs"]:
                raise ValueError("dynamic_queries=True requires 'num_truth_particles' in inputs")
            num_queries = int(x["inputs"]["num_truth_particles"].item())
            x["query_embed"] = torch.zeros(batch_size, num_queries, self.dim, device=x["key_embed"].device, dtype=x["key_embed"].dtype)
            x["query_valid"] = torch.full((batch_size, num_queries), True, device=x["key_embed"].device)
        else:
            # Standard learnable queries
            x["query_embed"] = self.initial_queries.expand(batch_size, -1, -1)
            x["query_valid"] = torch.full((batch_size, self.num_queries), True, device=x["query_embed"].device)

        if self.posenc:
            x["query_posenc"], x["key_posenc"] = self.generate_positional_encodings(x)

        attn_mask_lca = attn_mask_lca_transpose = attn_bias_lca = attn_bias_lca_transpose = None
        attn_mask_lca_first = attn_mask_lca_first_transpose = None
        attn_bias_lca_first = attn_bias_lca_first_transpose = None
        query_phi_for_lca = key_phi_for_lca = None
        if self.local_strided_mask_from_phi:
            if "query_phi" not in x or "key_phi" not in x:
                raise ValueError("query_phi and key_phi are required when local_strided_mask_from_phi is True")
            query_phi_for_lca = x["query_phi"]
            key_phi_for_lca = x["key_phi"]
        if self.local_strided_attn:
            assert x["query_embed"].shape[0] == 1, "Local strided attention only supports batch size 1"
            attn_mask_lca, attn_mask_lca_transpose, attn_bias_lca, attn_bias_lca_transpose = self.build_local_strided_artifacts(
                x["query_embed"],
                x["key_embed"],
                self.window_size,
                self.local_strided_decay_tau if self.local_strided_decay else None,
                query_phi=query_phi_for_lca,
                key_phi=key_phi_for_lca,
            )
            if self.local_strided_first_layer_window_size is not None:
                first_tau = (
                    self.local_strided_first_layer_decay_tau if self.local_strided_first_layer_decay_tau is not None else self.local_strided_decay_tau
                )
                (
                    attn_mask_lca_first,
                    attn_mask_lca_first_transpose,
                    attn_bias_lca_first,
                    attn_bias_lca_first_transpose,
                ) = self.build_local_strided_artifacts(
                    x["query_embed"],
                    x["key_embed"],
                    self.local_strided_first_layer_window_size,
                    first_tau if self.local_strided_decay else None,
                    query_phi=query_phi_for_lca,
                    key_phi=key_phi_for_lca,
                )

        if self.add_pe_pre_layer:
            x["query_embed"] = x["query_embed"] + x["query_posenc"]
            x["key_embed"] = x["key_embed"] + x["key_posenc"]

        outputs: dict[str, dict] = {}
        apply_query_masks = self.use_query_masks and not self.log_query_masks_only
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            outputs[f"layer_{layer_index}"] = {}
            if self.log_query_phi and "query_phi" in x:
                outputs[f"layer_{layer_index}"]["query_phi"] = x["query_phi"].detach().clone()
            if self.log_key_phi and "key_phi" in x:
                outputs[f"layer_{layer_index}"]["key_phi"] = x["key_phi"].detach().clone()
            if not self.mask_attention:
                layer_uses_mask_attention = False
            elif self.mask_attention_first_layer_only:
                layer_uses_mask_attention = layer_index == 0
            elif self.mask_attention_num_layers is not None:
                layer_uses_mask_attention = layer_index < self.mask_attention_num_layers
            else:
                layer_uses_mask_attention = layer_index >= self.mask_attention_start_layer
            
            if self.local_strided_first_layer_only:
                layer_uses_local_strided = self.local_strided_attn and (layer_index == 0)
            else:
                layer_uses_local_strided = self.local_strided_attn and (layer_index >= self.local_strided_start_layer)
            layer_uses_phi_distance_mask = self.use_phi_distance_mask

            # if maskattention, PE should be added before generating the mask
            if self.posenc and self.mask_attention and self.add_direct_pe_in_layer:
                x["query_embed"] = x["query_embed"] + x["query_posenc"]
                x["key_embed"] = x["key_embed"] + x["key_posenc"]

            if self.use_first_layer_mask_only:
                compute_attn_mask = layer_index == 0
            elif self.every_other_layer:
                compute_attn_mask = not (layer_index % 2 == 0)
            else:
                compute_attn_mask = True

            apply_task_mask = compute_attn_mask and layer_uses_mask_attention
            collect_task_masks = compute_attn_mask and (layer_uses_mask_attention or layer_uses_local_strided or layer_uses_phi_distance_mask)

            attn_masks: dict[str, torch.Tensor] = {}
            use_task_masks = collect_task_masks
            query_mask = None
            layer_attn_mask: torch.Tensor | None = None
            layer_attn_mask_transpose = None
            layer_attn_bias: torch.Tensor | None = None
            layer_attn_bias_transpose: torch.Tensor | None = None
            kv_attn_mask_for_logging: torch.Tensor | None = None

            assert self.tasks is not None
            for task in self.tasks:
                if not task.has_intermediate_loss:
                    continue
                if layer_index == 0 and not task.has_first_layer_loss:
                    continue

                # Get the outputs of the task given the current embeddings
                task_outputs = task(x)

                # Update x with task outputs for downstream use
                if isinstance(task, IncidenceRegressionTask):
                    x["incidence"] = task_outputs[task.incidence_key].detach()

                outputs[f"layer_{layer_index}"][task.name] = task_outputs

                if use_task_masks:
                    # Collect attention masks from different tasks
                    task_attn_masks = task.attn_mask(task_outputs)
                    for input_name, task_attn_mask in task_attn_masks.items():
                        if input_name in attn_masks:
                            attn_masks[input_name] |= task_attn_mask
                        else:
                            attn_masks[input_name] = task_attn_mask

                if isinstance(task, ObjectClassificationTask):
                    task_query_mask = task.query_mask(task_outputs)
                    if task_query_mask is not None:
                        query_mask = task_query_mask if query_mask is None else query_mask | task_query_mask
                        if apply_query_masks:
                            x["query_mask"] = query_mask
                        # query_mask is collected above and will be stored in outputs later
                        # regardless of whether we're applying it, as long as log_query_masks_only is True

            # Store query_mask in outputs if it was created
            # Store it whenever it's created, regardless of layer index
            # This allows logging callbacks to access it
            if query_mask is not None:
                outputs[f"layer_{layer_index}"]["query_mask"] = query_mask.detach().clone()

            task_attn_mask: torch.Tensor | None = None
            if attn_masks:
                if self.unified_decoding:
                    if len(attn_masks) > 1:
                        raise ValueError(f"In merged input mode, expected only one attention mask, got {len(attn_masks)}")
                    task_attn_mask = next(iter(attn_masks.values()))
                    if task_attn_mask.dim() == 2:
                        task_attn_mask = task_attn_mask.unsqueeze(-1).expand(-1, -1, num_constituents)
                else:
                    num_queries_actual = x["query_embed"].shape[1]
                    task_attn_mask = torch.full((batch_size, num_queries_actual, num_constituents), False, device=x["key_embed"].device)
                    for input_name, mask in attn_masks.items():
                        task_mask = mask.flatten()
                        task_attn_mask[x[f"key_is_{input_name}"].unsqueeze(1).expand_as(task_attn_mask)] = task_mask

                task_attn_mask = task_attn_mask.detach()
                outputs[f"layer_{layer_index}"]["task_attn_mask"] = task_attn_mask.clone()
                if apply_task_mask:
                    outputs[f"layer_{layer_index}"]["attn_mask"] = task_attn_mask

                    task_attn_mask_for_layer = task_attn_mask
                    apply_no_if_none = not self.no_if_none_then_all
                    if self.no_if_none_then_all_first_layer_only and layer_index > 0:
                        apply_no_if_none = False
                    if apply_no_if_none:
                        task_attn_mask_for_layer = torch.where(
                            torch.all(~task_attn_mask_for_layer, dim=-1, keepdim=True), True, task_attn_mask_for_layer
                        )
                    layer_attn_mask = task_attn_mask_for_layer

            if self.use_phi_distance_mask:
                layer_attn_mask = self.compute_phi_distance_mask(x)
                outputs[f"layer_{layer_index}"]["phi_distance_mask"] = layer_attn_mask.detach().clone()

            current_lca_mask = current_lca_mask_transpose = current_lca_bias = current_lca_bias_transpose = None
            if layer_uses_local_strided:
                if layer_index == 0 and attn_mask_lca_first is not None:
                    current_lca_mask = attn_mask_lca_first
                    current_lca_mask_transpose = attn_mask_lca_first_transpose
                    current_lca_bias = attn_bias_lca_first
                    current_lca_bias_transpose = attn_bias_lca_first_transpose
                else:
                    current_lca_mask = attn_mask_lca
                    current_lca_mask_transpose = attn_mask_lca_transpose
                    current_lca_bias = attn_bias_lca
                    current_lca_bias_transpose = attn_bias_lca_transpose

            if layer_uses_local_strided and current_lca_mask is not None and torch.is_tensor(current_lca_mask):
                outputs[f"layer_{layer_index}"]["lca_mask"] = current_lca_mask.detach().clone()

            if layer_uses_local_strided and current_lca_mask is not None:
                if layer_attn_mask is not None:
                    if not self.combine_ma_lca:
                        raise AssertionError(
                            "combine_ma_lca must be provided when both mask_attention and local_strided_attn are active on the same decoder layer"
                        )
                    if not (torch.is_tensor(layer_attn_mask) and torch.is_tensor(current_lca_mask)):
                        raise ValueError("combine_ma_lca currently requires tensor attention masks")
                    if self.combine_ma_lca == "OR":
                        layer_attn_mask = layer_attn_mask | current_lca_mask
                    else:
                        layer_attn_mask = layer_attn_mask & current_lca_mask
                else:
                    layer_attn_mask = current_lca_mask
                    if not torch.is_tensor(current_lca_mask):
                        layer_attn_mask_transpose = current_lca_mask_transpose
                if current_lca_bias is not None and torch.is_tensor(current_lca_bias):
                    layer_attn_bias = current_lca_bias
                    layer_attn_bias_transpose = current_lca_bias_transpose

            if layer_attn_mask is not None and f"layer_{layer_index}" in outputs:
                if "attn_mask" not in outputs[f"layer_{layer_index}"]:
                    outputs[f"layer_{layer_index}"]["attn_mask"] = (
                        layer_attn_mask.detach().clone() if torch.is_tensor(layer_attn_mask) else layer_attn_mask
                    )
            if layer_attn_bias is not None and f"layer_{layer_index}" in outputs and torch.is_tensor(layer_attn_bias):
                outputs[f"layer_{layer_index}"]["attn_bias"] = layer_attn_bias.detach().clone()

            if layer_attn_mask is not None and decoder_layer.bidirectional_ca and torch.is_tensor(layer_attn_mask) and self.attn_type != "flex":
                kv_attn_mask_for_logging = layer_attn_mask.transpose(-2, -1).detach().clone()

            if (layer_attn_mask is not None) and (self.attn_type == "flex") and torch.is_tensor(layer_attn_mask):
                outputs[f"layer_{layer_index}"]["attn_mask"] = layer_attn_mask

                B, Q_LEN, KV_LEN = layer_attn_mask.shape
                H = 1

                allowed_mask = (layer_attn_mask == 1).unsqueeze(1)

                def mask_mod(b, h, q_idx, kv_idx):
                    return allowed_mask[b, h, q_idx, kv_idx]

                def mask_mod_t(b, h, q_idx, kv_idx):
                    return mask_mod(b, h, kv_idx, q_idx)

                if self.bidirectional_ca:
                    layer_attn_mask_transpose = create_block_mask(mask_mod_t, B=B, H=H, Q_LEN=KV_LEN, KV_LEN=Q_LEN, device=layer_attn_mask.device)

                layer_attn_mask = create_block_mask(mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=layer_attn_mask.device)

            if layer_attn_bias is not None and layer_attn_bias_transpose is None and torch.is_tensor(layer_attn_bias):
                layer_attn_bias_transpose = layer_attn_bias.transpose(-2, -1)

            # Compute bidirectional mask using the configured transpose mode (for torch attention with tensor masks)
            if (
                self.bidirectional_ca_transpose_mode != "simple"
                and layer_attn_mask is not None
                and torch.is_tensor(layer_attn_mask)
                and self.attn_type != "flex"
            ):
                layer_attn_mask_transpose = self.compute_bidirectional_mask(
                    layer_attn_mask,
                    x.get("query_phi"),
                    x.get("key_phi"),
                    mode=self.bidirectional_ca_transpose_mode,
                )
                # Update logging mask for the bidirectional direction
                if decoder_layer.bidirectional_ca:
                    kv_attn_mask_for_logging = layer_attn_mask_transpose.detach().clone()

            # Apply window narrowing for bidirectional CA
            # Option 1: Hard narrowing (scale < 1.0, soft_narrowing=False)
            # Option 2: Soft narrowing via attention bias (soft_narrowing=True)
            if (
                self.bidirectional_ca_soft_narrowing
                and layer_attn_mask_transpose is not None
                and torch.is_tensor(layer_attn_mask_transpose)
            ):
                # Soft narrowing: compute attention bias that downweights distant queries
                # This preserves gradients and allows attending to "true" owner even if distant
                soft_bias = self.compute_soft_narrowing_bias(
                    layer_attn_mask_transpose,
                    self.bidirectional_ca_narrowing_tau,
                    query_phi=x.get("query_phi"),
                    key_phi=x.get("key_phi"),
                )
                # Add to existing transpose bias or create new
                if layer_attn_bias_transpose is not None:
                    layer_attn_bias_transpose = layer_attn_bias_transpose + soft_bias
                else:
                    layer_attn_bias_transpose = soft_bias

            elif (
                self.bidirectional_ca_window_scale < 1.0
                and layer_attn_mask_transpose is not None
                and torch.is_tensor(layer_attn_mask_transpose)
            ):
                # Hard narrowing: completely cut off distant queries
                layer_attn_mask_transpose = self.narrow_bidirectional_mask(
                    layer_attn_mask_transpose,
                    self.bidirectional_ca_window_scale,
                    query_phi=x.get("query_phi"),
                    key_phi=x.get("key_phi"),
                )
                # Update logging mask with the narrowed version
                if decoder_layer.bidirectional_ca:
                    kv_attn_mask_for_logging = layer_attn_mask_transpose.detach().clone()

            # Determine per-layer bidirectional CA control
            layer_use_bidirectional = layer_index >= self.bidirectional_ca_start_layer

            # Update the keys and queries
            q_mask = x.get("query_mask") if apply_query_masks else None
            x["query_embed"], x["key_embed"] = decoder_layer(
                x["query_embed"],
                x["key_embed"],
                attn_mask=layer_attn_mask,
                attn_bias=layer_attn_bias,
                q_mask=q_mask,
                kv_mask=x.get("key_valid"),
                query_posenc=x["query_posenc"] if (self.posenc and self.add_pe_in_attn) else None,
                key_posenc=x["key_posenc"] if (self.posenc and self.add_pe_in_attn) else None,
                attn_mask_transpose=layer_attn_mask_transpose,
                attn_bias_transpose=layer_attn_bias_transpose,
                use_bidirectional_ca=layer_use_bidirectional,
                use_key_values=self.bidirectional_ca_use_key_values,
            )

            if kv_attn_mask_for_logging is not None:
                outputs[f"layer_{layer_index}"]["attn_mask_kv"] = kv_attn_mask_for_logging
            if decoder_layer.last_q_sa_delta is not None:
                outputs[f"layer_{layer_index}"]["q_sa_delta"] = decoder_layer.last_q_sa_delta.detach().clone()

            # update the individual input constituent representations only if not in merged input mode
            if not self.unified_decoding:
                x = unmerge_inputs(x, input_names)

        return x, outputs

    def flex_local_ca_mask(
        self,
        q_len: int,
        kv_len: int,
        device,
        window_size: int | None = None,
        dtype_float: torch.dtype | None = None,  # noqa: ARG002 - maintained for compatibility
    ):
        # Calculate stride based on the ratio of key length to query length
        kv_len = torch.tensor(kv_len, device=device)
        window = window_size if window_size is not None else self.window_size
        window_mask_func = sliding_window_mask_strided_wrapped if self.window_wrap else sliding_window_mask_strided
        return window_mask_func(window, q_len=q_len, kv_len=kv_len, device=str(device))

    def build_local_strided_artifacts(
        self,
        q_embed: torch.Tensor,
        key_embed: torch.Tensor,
        window_size: int,
        decay_tau: float | None,
        query_phi: torch.Tensor | None = None,
        key_phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        attn_mask = None
        attn_mask_transpose = None
        attn_bias = None
        attn_bias_transpose = None
        if (self.attn_type == "torch") or (self.mask_attention):
            attn_mask = auto_local_ca_mask(
                q_embed,
                key_embed,
                window_size,
                wrap=self.window_wrap,
                flipped=self.flipped,
                query_phi=query_phi,
                key_phi=key_phi,
                shift_absolute=self.lca_shift_absolute,
                shift_fractional=self.lca_shift_fractional,
                shift_queries=self.lca_shift_queries,
            )
        elif self.attn_type == "flex":
            device = q_embed.device
            q_len = q_embed.shape[1]
            kv_len = key_embed.shape[1]
            dtype_float = q_embed.dtype
            if self.fast_local_ca:
                attn_mask = build_strided_sliding_window_blockmask(
                    window_size=window_size,
                    block_size=self.block_size,
                    q_len=q_len,
                    kv_len=kv_len,
                    device=device,
                    wrap=self.window_wrap,
                    dtype_float=dtype_float,
                )
            else:
                attn_mask = self.flex_local_ca_mask(q_len, kv_len, device, window_size=window_size, dtype_float=dtype_float)
            if self.bidirectional_ca:
                attn_mask_transpose = transpose_blockmask(attn_mask, q_tokens=q_len, kv_tokens=kv_len, dev=device)

        if self.local_strided_decay and decay_tau is not None and attn_mask is not None and torch.is_tensor(attn_mask):
            attn_bias = self.compute_local_strided_decay_bias(attn_mask, q_embed.dtype, decay_tau)
            attn_bias_transpose = attn_bias.transpose(-2, -1)

        return attn_mask, attn_mask_transpose, attn_bias, attn_bias_transpose

    def compute_local_strided_decay_bias(self, attn_mask_lca: torch.Tensor, dtype: torch.dtype, tau: float | None) -> torch.Tensor:
        """Build an attention bias tensor with exponential decay inside the local CA window."""
        assert attn_mask_lca.dim() == 3, "Expected attention mask with shape (B, Q, K)"
        batch_size, q_len, kv_len = attn_mask_lca.shape
        assert batch_size == 1, "local_strided_attn currently only supports batch size 1"
        if tau is None:
            tau = self.local_strided_decay_tau

        device = attn_mask_lca.device
        stride = kv_len / q_len
        query_indices = torch.arange(q_len, device=device, dtype=dtype)
        if self.flipped:
            query_indices = torch.flip(query_indices, dims=(0,))
        centers = torch.round(query_indices * stride)
        key_positions = torch.arange(kv_len, device=device, dtype=dtype)
        deltas = torch.abs(key_positions.unsqueeze(0) - centers.unsqueeze(1))
        if self.window_wrap:
            wrap_extent = torch.tensor(float(kv_len), device=device, dtype=dtype)
            deltas = torch.minimum(deltas, wrap_extent - deltas)

        weights = torch.exp(-deltas / tau)
        weights = weights.unsqueeze(0)
        weights = weights * attn_mask_lca.to(dtype)
        weights = torch.clamp(weights, min=1e-6)
        return torch.log(weights)

    def generate_positional_encodings(self, x: dict):
        device = x["query_embed"].device
        dtype = x["query_embed"].dtype
        batch_size = x["query_embed"].shape[0]
        num_queries = x["query_embed"].shape[1]  # Use actual query count (supports dynamic queries)

        idx = torch.arange(num_queries, device=device, dtype=dtype)
        query_fraction = idx / max(num_queries, 1)
        phi_shift = self.phi_shift.to(device=device, dtype=dtype)
        default_query_phi = 2 * torch.pi * (query_fraction - phi_shift - 0.5)

        if self.phi_shift_from_key_phi:
            if "key_phi" not in x:
                raise ValueError("key_phi is required when phi_shift_from_key_phi is True")
            min_key_phi = torch.amin(x["key_phi"], dim=-1, keepdim=True)
            denom = max(num_queries - 1, 1)
            span_fraction = idx / denom
            x["query_phi"] = min_key_phi + 2 * torch.pi * span_fraction
        elif self.quantile_query_phi:
            if "key_phi" not in x:
                raise ValueError("key_phi is required when quantile_query_phi is True")
            x["query_phi"] = self._compute_quantile_query_phi(x, default_query_phi, device, dtype)
        elif self.histogram_query_phi:
            if "key_phi" not in x:
                raise ValueError("key_phi is required when histogram_query_phi is True")
            x["query_phi"] = self._compute_histogram_query_phi(x, default_query_phi, device, dtype)
        else:
            x["query_phi"] = default_query_phi.expand(batch_size, -1)

        query_posenc = pos_enc_symmetric(x["query_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        key_posenc = pos_enc_symmetric(x["key_phi"], self.dim, self.posenc["alpha"], self.posenc["base"])
        return query_posenc, key_posenc

    def _collect_valid_key_phi(self, x: dict) -> list[Tensor]:
        key_phi = x["key_phi"]
        key_valid = x.get("key_valid")
        if key_valid is not None and key_valid.dtype != torch.bool:
            key_valid = key_valid.to(torch.bool)

        valid_key_phi: list[Tensor] = []
        for batch_idx in range(key_phi.shape[0]):
            if key_valid is None:
                valid_phi = key_phi[batch_idx]
            else:
                mask = key_valid[batch_idx]
                valid_phi = key_phi[batch_idx][mask]
                if valid_phi.numel() == 0:
                    valid_phi = key_phi[batch_idx]
            valid_key_phi.append(valid_phi)
        return valid_key_phi

    def _compute_quantile_query_phi(self, x: dict, default_query_phi: Tensor, device, dtype) -> Tensor:
        valid_key_phi = self._collect_valid_key_phi(x)
        num_queries = x["query_embed"].shape[1]  # Use actual query count (supports dynamic queries)
        query_phi = torch.empty((len(valid_key_phi), num_queries), device=device, dtype=dtype)
        fractions = torch.linspace(0.0, 1.0, steps=num_queries, device=device, dtype=dtype)

        for batch_idx, hits in enumerate(valid_key_phi):
            if hits.numel() == 0:
                query_phi[batch_idx] = default_query_phi
                continue
            if hits.numel() == 1:
                query_phi[batch_idx].fill_(hits[0])
                continue

            sorted_phi = torch.sort(hits).values
            max_index = sorted_phi.shape[0] - 1
            scaled = fractions * max_index
            lower = torch.floor(scaled).long()
            upper = torch.clamp(lower + 1, max=max_index)
            frac = scaled - lower.to(dtype)
            query_phi[batch_idx] = sorted_phi[lower] * (1 - frac) + sorted_phi[upper] * frac

        return query_phi

    def _compute_histogram_query_phi(self, x: dict, default_query_phi: Tensor, device, dtype) -> Tensor:
        valid_key_phi = self._collect_valid_key_phi(x)
        num_queries = x["query_embed"].shape[1]  # Use actual query count (supports dynamic queries)
        query_phi = torch.empty((len(valid_key_phi), num_queries), device=device, dtype=dtype)
        hist_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        bin_edges = torch.linspace(-torch.pi, torch.pi, self.histogram_query_phi_bins + 1, device=device, dtype=hist_dtype)
        bin_centers = (0.5 * (bin_edges[:-1] + bin_edges[1:])).to(dtype)
        fractions = torch.linspace(0.0, 1.0, steps=num_queries, device=device, dtype=hist_dtype)

        bucket_boundaries = bin_edges[1:-1]
        for batch_idx, hits in enumerate(valid_key_phi):
            if hits.numel() == 0:
                query_phi[batch_idx] = default_query_phi
                continue

            hits_hist = hits.to(hist_dtype)
            if bucket_boundaries.numel() == 0:
                bin_ids = torch.zeros_like(hits_hist, dtype=torch.long)
            else:
                bin_ids = torch.bucketize(hits_hist, bucket_boundaries)
            bin_ids = bin_ids.clamp(0, self.histogram_query_phi_bins - 1)

            counts = torch.zeros(self.histogram_query_phi_bins, device=device, dtype=hist_dtype)
            counts.scatter_add_(0, bin_ids, torch.ones_like(hits_hist, dtype=hist_dtype))
            total = counts.sum()
            if total <= 0:
                query_phi[batch_idx] = default_query_phi
                continue

            cdf = torch.cumsum(counts, dim=0)
            cdf = torch.cat([torch.zeros(1, device=device, dtype=hist_dtype), cdf]) / total
            bin_indices = torch.searchsorted(cdf, fractions, right=False) - 1
            bin_indices = bin_indices.clamp(0, self.histogram_query_phi_bins - 1)
            query_phi[batch_idx] = bin_centers[bin_indices]

        return query_phi

    def compute_phi_distance_mask(self, x: dict) -> torch.Tensor:
        if "query_phi" not in x or "key_phi" not in x:
            raise ValueError("query_phi and key_phi are required to compute phi distance mask")
        query_phi = x["query_phi"]
        key_phi = x["key_phi"]
        diff = query_phi.unsqueeze(-1) - key_phi.unsqueeze(-2)
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        threshold = self.phi_distance_threshold
        if threshold <= 0:
            threshold = self._auto_phi_distance_threshold(key_phi)
        mask = torch.abs(diff) <= threshold
        return mask

    def _auto_phi_distance_threshold(self, key_phi: torch.Tensor) -> torch.Tensor:
        key_len = key_phi.shape[-1]
        if key_len <= 0:
            return torch.tensor(0.0, device=key_phi.device, dtype=key_phi.dtype)
        avg_spacing = (2 * torch.pi) / key_len
        half_window = self.window_size / 2 if self.window_size > 0 else key_len / 2
        threshold = avg_spacing * half_window
        return torch.tensor(float(threshold), device=key_phi.device, dtype=key_phi.dtype)

    def compute_phi_aware_transpose(self, mask: torch.Tensor, query_phi: torch.Tensor, key_phi: torch.Tensor) -> torch.Tensor:
        """Compute a φ-aware transpose of the attention mask.

        Instead of simple index transpose, this computes which query each key should attend to
        based on φ similarity. For each key at φ_k, find the query with the closest φ_q,
        and use that query's attention pattern.

        Args:
            mask: Attention mask of shape (B, Q, K) where mask[b, i, j] indicates
                  query i attends to key j.
            query_phi: Query phi values of shape (B, Q).
            key_phi: Key phi values of shape (B, K).

        Returns:
            Transposed mask of shape (B, K, Q) suitable for bidirectional CA where
            keys attend to queries.
        """
        B, Q, K = mask.shape
        device = mask.device
        dtype = query_phi.dtype

        # For each key position, find the query with closest phi (accounting for wrapping)
        # key_phi: (B, K), query_phi: (B, Q)
        # Compute wrapped phi difference: key_phi[b, k] - query_phi[b, q]
        phi_diff = key_phi.unsqueeze(-1) - query_phi.unsqueeze(-2)  # (B, K, Q)
        phi_diff = (phi_diff + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-π, π]

        # Find the query with minimum absolute phi difference for each key
        closest_query_idx = torch.argmin(torch.abs(phi_diff), dim=-1)  # (B, K)

        # Gather the attention patterns: for key k, use mask[closest_query[k], :]
        # We want: transposed_mask[b, k, q] = mask[b, closest_query[b, k], ?]
        # But we need to think about what the transposed mask represents:
        # - Original mask[b, i, j] = 1 means query i attends to key j
        # - Transposed mask[b, k, q] = 1 means key k attends to query q
        #
        # With φ-aware transpose:
        # - Key k should attend to queries that are "similar" to it in φ-space
        # - The query i* with closest φ to key k represents "what a query at this φ does"
        # - So key k inherits the attention pattern of query i*
        # - But transposed: key k attends to the queries that would attend to a key at similar φ
        #
        # Implementation: transposed_mask[b, k, :] = mask[b, closest_query[b, k], :] transposed
        # Actually, we want: for key k at φ_k, which queries should it attend to?
        # Answer: queries whose φ is close to φ_k (same logic as forward direction)
        #
        # Simpler approach: just reindex then transpose
        # new_mask[b, k, j] = mask[b, closest_query[b, k], j]
        # This says: key k attends to key j if the query at similar-φ-to-k attended to j
        # Then transpose to get key→query pattern

        # Gather mask patterns based on closest query
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)  # (B, K)
        # reindexed_mask[b, k, j] = mask[b, closest_query[b, k], j]
        reindexed_mask = mask[batch_idx, closest_query_idx, :]  # (B, K, K)

        # Now we have key→key pattern. We need key→query pattern.
        # The issue is that mask is Q×K, and we reindexed to get K×K.
        # We need to map this back to K×Q.

        # Alternative approach: directly compute key→query based on φ similarity
        # For key k, it should attend to query q if φ_k ≈ φ_q
        # This is just the transpose of the φ-similarity logic

        # Actually, let's reconsider. The original mask encodes:
        # "query i should attend to key j based on learned + positional factors"
        # For bidirectional, we want:
        # "key j should attend to query i based on learned + positional factors"
        #
        # The φ-aware insight: if query i* has similar φ to key j, then key j
        # should attend to the same queries that attended to keys at i*'s φ.
        #
        # So: transposed[j, :] ≈ mask[i*, :].T where i* = argmin |φ_q[i*] - φ_k[j]|
        #
        # But mask[i*, :] is over keys, not queries. We need to think more carefully.
        #
        # Final approach: for each (key_j, query_i) pair:
        # - Find the query i* closest in φ to key_j
        # - Find the key j* closest in φ to query_i
        # - transposed[j, i] = mask[i*, j*]
        # This maps "would a query at key_j's φ attend to a key at query_i's φ?"

        # Compute key index closest to each query's phi
        phi_diff_kq = query_phi.unsqueeze(-1) - key_phi.unsqueeze(-2)  # (B, Q, K)
        phi_diff_kq = (phi_diff_kq + torch.pi) % (2 * torch.pi) - torch.pi
        closest_key_idx = torch.argmin(torch.abs(phi_diff_kq), dim=-1)  # (B, Q)

        # Now construct the transposed mask
        # transposed[b, k, q] = mask[b, closest_query[b, k], closest_key[b, q]]
        transposed_mask = torch.zeros(B, K, Q, device=device, dtype=mask.dtype)

        for b in range(B):
            for k in range(K):
                i_star = closest_query_idx[b, k]
                for q in range(Q):
                    j_star = closest_key_idx[b, q]
                    transposed_mask[b, k, q] = mask[b, i_star, j_star]

        return transposed_mask

    def compute_phi_aware_transpose_vectorized(self, mask: torch.Tensor, query_phi: torch.Tensor, key_phi: torch.Tensor) -> torch.Tensor:
        """Vectorized version of compute_phi_aware_transpose.

        For each (key k, query q) pair in the transposed mask:
        - Find query i* with φ closest to key k's φ
        - Find key j* with φ closest to query q's φ
        - transposed[k, q] = mask[i*, j*]

        This asks: "would a query at key k's φ-position attend to a key at query q's φ-position?"
        """
        B, Q, K = mask.shape
        device = mask.device

        # Compute phi differences with wrapping
        # For each key, find closest query
        phi_diff_kq = key_phi.unsqueeze(-1) - query_phi.unsqueeze(-2)  # (B, K, Q)
        phi_diff_kq = (phi_diff_kq + torch.pi) % (2 * torch.pi) - torch.pi
        closest_query_for_key = torch.argmin(torch.abs(phi_diff_kq), dim=-1)  # (B, K)

        # For each query, find closest key
        phi_diff_qk = query_phi.unsqueeze(-1) - key_phi.unsqueeze(-2)  # (B, Q, K)
        phi_diff_qk = (phi_diff_qk + torch.pi) % (2 * torch.pi) - torch.pi
        closest_key_for_query = torch.argmin(torch.abs(phi_diff_qk), dim=-1)  # (B, Q)

        # Build index tensors for gathering
        # We want transposed[b, k, q] = mask[b, closest_query_for_key[b, k], closest_key_for_query[b, q]]
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, Q)
        k_to_q_idx = closest_query_for_key.unsqueeze(-1).expand(B, K, Q)  # (B, K, Q)
        q_to_k_idx = closest_key_for_query.unsqueeze(-2).expand(B, K, Q)  # (B, K, Q)

        # Gather from mask using advanced indexing
        transposed_mask = mask[batch_idx, k_to_q_idx, q_to_k_idx]

        return transposed_mask

    def compute_direct_phi_mask(self, query_phi: torch.Tensor, key_phi: torch.Tensor, window_fraction: float = 0.1) -> torch.Tensor:
        """Compute a K×Q attention mask directly from φ similarity.

        This ignores the task-predicted mask entirely and creates a mask based purely
        on φ proximity between keys and queries. Useful for bidirectional CA where
        we want keys to attend to queries with similar φ.

        Args:
            query_phi: Query phi values of shape (B, Q).
            key_phi: Key phi values of shape (B, K).
            window_fraction: Fraction of 2π to use as the attention window (default 0.1 = ~36°).

        Returns:
            Mask of shape (B, K, Q) where mask[b, k, q] = 1 if |φ_k - φ_q| < threshold.
        """
        # Compute wrapped phi difference: key_phi[b, k] - query_phi[b, q]
        phi_diff = key_phi.unsqueeze(-1) - query_phi.unsqueeze(-2)  # (B, K, Q)
        phi_diff = (phi_diff + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-π, π]

        # Threshold based on window fraction
        threshold = window_fraction * torch.pi  # window_fraction of π radians on each side

        mask = torch.abs(phi_diff) <= threshold
        return mask

    def compute_scaled_transpose(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute a scaled transpose that accounts for different Q and K dimensions.

        When Q ≠ K, the simple transpose changes the aspect ratio of the diagonal.
        This method rescales the indices so that the diagonal in K×Q space
        corresponds to the same relative positions as in Q×K space.

        For each (k, q) position in the output K×Q mask:
        - Map k to the equivalent query position: i = k * Q / K
        - Map q to the equivalent key position: j = q * K / Q
        - Look up mask[round(i), round(j)]

        Args:
            mask: Attention mask of shape (B, Q, K).

        Returns:
            Transposed mask of shape (B, K, Q) with scaled indices.
        """
        B, Q, K = mask.shape
        device = mask.device

        # Create index grids for K×Q output
        k_indices = torch.arange(K, device=device, dtype=torch.float32)
        q_indices = torch.arange(Q, device=device, dtype=torch.float32)

        # Map k to equivalent query index: i = k * Q / K
        # Map q to equivalent key index: j = q * K / Q
        i_from_k = (k_indices * Q / K).round().long().clamp(0, Q - 1)  # (K,)
        j_from_q = (q_indices * K / Q).round().long().clamp(0, K - 1)  # (Q,)

        # Build index tensors for gathering
        # transposed[b, k, q] = mask[b, i_from_k[k], j_from_q[q]]
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, K, Q)
        i_idx = i_from_k.view(1, K, 1).expand(B, K, Q)
        j_idx = j_from_q.view(1, 1, Q).expand(B, K, Q)

        transposed_mask = mask[batch_idx, i_idx, j_idx]
        return transposed_mask

    def compute_bidirectional_mask(
        self,
        mask: torch.Tensor,
        query_phi: torch.Tensor | None,
        key_phi: torch.Tensor | None,
        mode: str = "simple",
    ) -> torch.Tensor:
        """Compute the attention mask for bidirectional CA based on the specified mode.

        Args:
            mask: Original attention mask of shape (B, Q, K).
            query_phi: Query phi values of shape (B, Q), required for some modes.
            key_phi: Key phi values of shape (B, K), required for some modes.
            mode: Transpose mode - "simple", "phi_aware", "direct_phi", or "scaled".

        Returns:
            Transposed/transformed mask of shape (B, K, Q) for bidirectional CA.
        """
        if mode == "simple":
            return mask.transpose(-2, -1)

        elif mode == "scaled":
            return self.compute_scaled_transpose(mask)

        elif mode == "phi_aware":
            if query_phi is None or key_phi is None:
                raise ValueError("phi_aware mode requires query_phi and key_phi")
            return self.compute_phi_aware_transpose_vectorized(mask, query_phi, key_phi)

        elif mode == "direct_phi":
            if query_phi is None or key_phi is None:
                raise ValueError("direct_phi mode requires query_phi and key_phi")
            # Use window size to determine the phi threshold
            Q, K = mask.shape[-2], mask.shape[-1]
            # Window fraction based on relative size: want similar coverage as the original mask
            window_fraction = 0.5 / max(Q, K) * self.window_size if self.window_size > 0 else 0.1
            window_fraction = min(window_fraction, 0.5)  # Cap at 90 degrees
            return self.compute_direct_phi_mask(query_phi, key_phi, window_fraction)

        else:
            raise ValueError(f"Unknown bidirectional_ca_transpose_mode: {mode}")

    def narrow_bidirectional_mask(
        self,
        mask: torch.Tensor,
        scale: float,
        query_phi: torch.Tensor | None = None,
        key_phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Narrow the bidirectional CA mask to reduce contamination from neighboring queries.

        Given a K×Q mask for bidirectional CA, this method narrows the attention window
        so that each key attends to fewer queries. The narrowing is done by keeping only
        the entries closest to the "center" of each key's attention pattern.

        Args:
            mask: Bidirectional attention mask of shape (B, K, Q).
            scale: Scale factor (0 < scale <= 1). E.g., scale=0.5 halves the window width.
            query_phi: Query phi values of shape (B, Q), used for φ-aware narrowing.
            key_phi: Key phi values of shape (B, K), used for φ-aware narrowing.

        Returns:
            Narrowed mask of shape (B, K, Q).
        """
        if scale >= 1.0:
            return mask

        B, K, Q = mask.shape
        device = mask.device

        if query_phi is not None and key_phi is not None:
            # φ-aware narrowing: keep entries where |φ_k - φ_q| is smallest
            # Compute wrapped phi difference
            phi_diff = key_phi.unsqueeze(-1) - query_phi.unsqueeze(-2)  # (B, K, Q)
            phi_diff = (phi_diff + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-π, π]
            phi_dist = torch.abs(phi_diff)

            # For each key, find the threshold distance that keeps scale fraction of entries
            # We use the mask to only consider currently allowed positions
            mask_bool = mask.bool()

            # Set distance to large value where mask is False
            large_value = torch.tensor(float('inf'), device=device)
            masked_dist = torch.where(mask_bool, phi_dist, large_value)

            # For each key (row), find the scale-th percentile distance among allowed queries
            # This determines the threshold for keeping entries
            narrowed_mask = torch.zeros_like(mask)

            for b in range(B):
                for k in range(K):
                    row_mask = mask_bool[b, k]
                    if not row_mask.any():
                        continue

                    row_dist = phi_dist[b, k]
                    allowed_dist = row_dist[row_mask]

                    # Keep only the closest (scale * count) entries
                    num_to_keep = max(1, int(allowed_dist.numel() * scale))
                    if num_to_keep >= allowed_dist.numel():
                        narrowed_mask[b, k] = mask[b, k]
                    else:
                        threshold = torch.kthvalue(allowed_dist, num_to_keep).values
                        keep = (row_dist <= threshold) & row_mask
                        narrowed_mask[b, k] = keep.to(mask.dtype)

            return narrowed_mask

        else:
            # Index-based narrowing: keep entries near the diagonal center
            # For each key k, the expected query position is q* = k * Q / K
            k_indices = torch.arange(K, device=device, dtype=torch.float32)
            q_indices = torch.arange(Q, device=device, dtype=torch.float32)

            # Expected query position for each key
            expected_q = k_indices * Q / K  # (K,)

            # Distance from expected position
            dist_from_center = torch.abs(q_indices.unsqueeze(0) - expected_q.unsqueeze(1))  # (K, Q)

            # For wrapped case, also consider wrap-around distance
            if self.window_wrap:
                wrap_dist = Q - dist_from_center
                dist_from_center = torch.minimum(dist_from_center, wrap_dist)

            # Original window width (estimate from mask)
            mask_bool = mask.bool()
            avg_width = mask_bool.float().sum(dim=-1).mean()  # average queries per key

            # New threshold
            new_half_width = (avg_width * scale) / 2

            # Keep entries within new half-width of center
            within_window = dist_from_center <= new_half_width
            within_window = within_window.unsqueeze(0).expand(B, -1, -1)

            narrowed_mask = mask_bool & within_window
            return narrowed_mask.to(mask.dtype)

    def compute_soft_narrowing_bias(
        self,
        mask: torch.Tensor,
        tau: float,
        query_phi: torch.Tensor | None = None,
        key_phi: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a soft attention bias that downweights distant queries without blocking them.

        Instead of a hard mask cutoff, this creates a log-space bias where distant queries
        get lower attention weight. This preserves gradients and allows the model to still
        attend to the "true" owning query even if it's outside the typical window.

        The bias is computed as: bias = -|φ_k - φ_q| / tau
        This means:
        - Queries at the same φ as the key get bias = 0 (no penalty)
        - Queries at distance d get bias = -d/tau (penalty proportional to distance)
        - After softmax, attention weights decay exponentially with distance

        Args:
            mask: Bidirectional attention mask of shape (B, K, Q).
            tau: Temperature parameter controlling decay rate. Smaller tau = sharper decay.
            query_phi: Query phi values of shape (B, Q).
            key_phi: Key phi values of shape (B, K).

        Returns:
            Attention bias of shape (B, K, Q) to be added to attention logits.
        """
        B, K, Q = mask.shape
        device = mask.device
        dtype = query_phi.dtype if query_phi is not None else torch.float32

        if query_phi is not None and key_phi is not None:
            # φ-aware soft narrowing
            phi_diff = key_phi.unsqueeze(-1) - query_phi.unsqueeze(-2)  # (B, K, Q)
            phi_diff = (phi_diff + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-π, π]
            phi_dist = torch.abs(phi_diff)  # (B, K, Q)

            # Normalize distance by π so tau has consistent meaning
            normalized_dist = phi_dist / torch.pi

            # Compute bias: closer queries get smaller penalty
            bias = -normalized_dist / tau

        else:
            # Index-based soft narrowing
            k_indices = torch.arange(K, device=device, dtype=dtype)
            q_indices = torch.arange(Q, device=device, dtype=dtype)

            # Expected query position for each key
            expected_q = k_indices * Q / K  # (K,)

            # Distance from expected position, normalized by Q
            dist = torch.abs(q_indices.unsqueeze(0) - expected_q.unsqueeze(1)) / Q  # (K, Q)

            # For wrapped case
            if self.window_wrap:
                wrap_dist = 1.0 - dist
                dist = torch.minimum(dist, wrap_dist)

            # Compute bias
            bias = -dist / tau
            bias = bias.unsqueeze(0).expand(B, -1, -1)

        # Apply mask: set bias to -inf where mask is False (completely block those positions)
        mask_bool = mask.bool()
        bias = torch.where(mask_bool, bias, torch.tensor(float('-inf'), device=device, dtype=dtype))

        return bias


class MaskFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: str = "LayerNorm",
        depth: int = 0,
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        bidirectional_ca: bool = True,
        enable_query_self_attn: bool = True,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        sa_pe: bool = False,
        scale_pe: float = 1.0,
    ) -> None:
        """Initialize a MaskFormer decoder layer.

        Args:
            dim: Embedding dimension.
            norm: Normalization type.
            depth: Layer depth index.
            dense_kwargs: Optional arguments for Dense layers.
            attn_kwargs: Optional arguments for Attention layers.
            bidirectional_ca: Enable bidirectional cross-attention.
            qkv_norm: Apply normalization to QKV in attention.
            hybrid_norm: Enable hybrid normalization from 2503.04598.
            sa_pe: Add PE pre self attention.
        """
        super().__init__()
        self.dim = dim
        self.bidirectional_ca = bidirectional_ca
        self.enable_query_self_attn = enable_query_self_attn
        self.sa_pe = sa_pe
        self.scale_pe = float(scale_pe)

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm)

        attn_kwargs = attn_kwargs or {}
        self.attn_type = attn_kwargs.get("attn_type", "torch")
        dense_kwargs = dense_kwargs or {}

        residual = partial(Residual, dim=dim)
        self.q_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_sa = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.q_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

        if self.bidirectional_ca:
            self.kv_ca = residual(Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
            self.kv_dense = residual(Dense(dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)
        self.last_q_sa_delta: Tensor | None = None

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        attn_mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        query_posenc: Tensor | None = None,
        key_posenc: Tensor | None = None,
        attn_mask_transpose: Tensor | None = None,
        attn_bias_transpose: Tensor | None = None,
        use_bidirectional_ca: bool | None = None,
        use_key_values: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the decoder layer.

        Args:
            q: Query embeddings.
            kv: Key/value embeddings.
            attn_mask: Optional attention mask.
            attn_bias: Optional attention bias tensor.
            q_mask: Optional query mask.
            kv_mask: Optional key/value mask.
            query_posenc: Optional query positional encoding.
            key_posenc: Optional key positional encoding.
            attn_mask_transpose: Optional transposed attention mask.
            attn_bias_transpose: Optional transposed attention bias tensor.
            use_bidirectional_ca: Override for bidirectional_ca (if None, uses self.bidirectional_ca).
            use_key_values: If True, use key embeddings as values in bidirectional CA instead of query embeddings.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - The updated query embeddings (Tensor).
                - The updated key/value embeddings (Tensor).
        """
        q_pe = q if query_posenc is None else q + self.scale_pe * query_posenc
        kv_pe = kv if key_posenc is None else kv + self.scale_pe * key_posenc

        q = self.q_ca(q_pe, k=kv_pe, v=kv, attn_mask=attn_mask, attn_bias=attn_bias, q_mask=q_mask, kv_mask=kv_mask)
        q = self.q_dense(q)

        q_before_sa = q
        if self.enable_query_self_attn:
            if self.sa_pe:
                q_pe = q if query_posenc is None else q + self.scale_pe * query_posenc
                q = self.q_sa(q_pe, k=q_pe, v=q, q_mask=q_mask)
            else:
                q = self.q_sa(q, k=q, v=q, q_mask=q_mask)
        q_after_sa = q
        self.last_q_sa_delta = (q_after_sa - q_before_sa).norm(dim=-1).detach()

        # Determine whether to use bidirectional CA for this layer
        do_bidirectional = use_bidirectional_ca if use_bidirectional_ca is not None else self.bidirectional_ca

        # Update key/constituent embeddings with the query/object embeddings
        if do_bidirectional and self.bidirectional_ca:  # self.bidirectional_ca ensures kv_ca exists
            if attn_mask is not None:
                if self.attn_type == "flex":
                    assert attn_mask_transpose is not None, "attn_mask_transpose must be provided for flex attention"
                # Index from the back so we are batch shape agnostic
                attn_mask = attn_mask_transpose if attn_mask_transpose is not None else attn_mask.transpose(-2, -1)
            if attn_bias is not None:
                if attn_bias_transpose is not None:
                    attn_bias = attn_bias_transpose
                else:
                    attn_bias = attn_bias.transpose(-2, -1)

            q_pe = q if query_posenc is None else q + self.scale_pe * query_posenc
            kv_pe = kv if key_posenc is None else kv + self.scale_pe * key_posenc

            # Choose value source: key embeddings (use_key_values=True) or query embeddings (default)
            v_for_bidi = kv if use_key_values else q

            kv = self.kv_ca(kv_pe, k=q_pe, v=v_for_bidi, attn_mask=attn_mask, attn_bias=attn_bias, q_mask=kv_mask, kv_mask=q_mask)
            kv = self.kv_dense(kv)

        return q, kv

    def set_backend(self, attn_type: str) -> None:
        """Set the backend for the attention layers.

        Args:
            attn_type: Attention implementation type to use.
        """
        self.q_ca.fn.set_backend(attn_type)
        self.q_sa.fn.set_backend(attn_type)

        if self.bidirectional_ca:
            self.kv_ca.fn.set_backend(attn_type)
