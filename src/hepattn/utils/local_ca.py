import torch


def get_local_ca_mask(
    n_objects,
    n_inputs,
    window_size,
    stride=1,
    device=None,
    wrap=False,
    shift_absolute: int | None = None,
    shift_fractional: float | None = None,
    shift_queries: int | None = None,
):
    assert window_size >= 0, "Window size must be non-negative"
    assert window_size % 2 == 0, "Window size must be even"
    shift_count = sum(x is not None for x in [shift_absolute, shift_fractional, shift_queries])
    assert shift_count <= 1, "Only one of shift_absolute, shift_fractional, or shift_queries can be set"

    # Compute the shift in key positions
    shift = 0
    if shift_absolute is not None:
        shift = shift_absolute
    elif shift_fractional is not None:
        shift = round(shift_fractional * n_inputs)
    elif shift_queries is not None:
        # Convert query positions to key positions by multiplying by stride
        shift = round(shift_queries * stride)

    mask = torch.zeros((n_objects, n_inputs), dtype=torch.bool, device=device)
    for i in range(n_objects):
        # Apply shift to the center position
        center = round(i * stride) + shift
        if wrap:
            center = center % n_inputs

        start_raw = center - window_size // 2
        start = max(0, start_raw)
        end_raw = center + window_size // 2 + 1
        end = min(n_inputs, end_raw)
        mask[i, start:end] = 1

        # if wrap, left and right ends are connected
        if wrap:
            if start_raw < 0:
                start = n_inputs + start_raw
                end = n_inputs
                mask[i, start:end] = 1
            if end_raw > n_inputs:
                start = 0
                end = end_raw - n_inputs
                mask[i, start:end] = 1

    return mask


def get_local_ca_mask_flipped(
    n_objects, n_inputs, window_size, stride=1, device=None, wrap=False
):
    assert window_size >= 0, "Window size must be non-negative"
    assert window_size % 2 == 0, "Window size must be even"

    mask = torch.zeros((n_objects, n_inputs), dtype=torch.bool, device=device)

    for i in range(n_objects):

        # reverse the order of rows to flip diagonal direction
        rev_i = n_objects - 1 - i

        start_raw = round(rev_i * stride) - window_size // 2
        end_raw   = round(rev_i * stride) + window_size // 2 + 1

        start = max(0, start_raw)
        end   = min(n_inputs, end_raw)
        mask[i, start:end] = 1

        if wrap:
            if start_raw < 0:
                mask[i, n_inputs + start_raw : n_inputs] = 1
            if end_raw > n_inputs:
                mask[i, 0 : end_raw - n_inputs] = 1

    return mask



def _window_mask_from_centers(
    centers,
    n_inputs,
    window_size,
    device,
    wrap,
    shift_absolute: int | None = None,
    shift_fractional: float | None = None,
    shift_queries: int | None = None,
    stride: float = 1.0,
):
    assert window_size >= 0, "Window size must be non-negative"
    assert window_size % 2 == 0, "Window size must be even"
    shift_count = sum(x is not None for x in [shift_absolute, shift_fractional, shift_queries])
    assert shift_count <= 1, "Only one of shift_absolute, shift_fractional, or shift_queries can be set"

    # Compute the shift in key positions
    shift = 0
    if shift_absolute is not None:
        shift = shift_absolute
    elif shift_fractional is not None:
        shift = round(shift_fractional * n_inputs)
    elif shift_queries is not None:
        shift = round(shift_queries * stride)

    mask = torch.zeros((centers.shape[0], n_inputs), dtype=torch.bool, device=device)
    half = window_size // 2
    for row, center in enumerate(centers.tolist()):
        # Apply shift to the center position
        shifted_center = int(round(center)) + shift
        if wrap:
            shifted_center = shifted_center % n_inputs

        start_raw = shifted_center - half
        end_raw = shifted_center + half + 1
        start = max(0, start_raw)
        end = min(n_inputs, end_raw)
        mask[row, start:end] = 1

        if wrap:
            if start_raw < 0:
                mask[row, n_inputs + start_raw : n_inputs] = 1
            if end_raw > n_inputs:
                mask[row, 0 : end_raw - n_inputs] = 1
    return mask


def _circular_distance(a: torch.Tensor, b: torch.Tensor, wrap: bool) -> torch.Tensor:
    diff = a.unsqueeze(-1) - b.unsqueeze(0)
    if wrap:
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.abs()


def auto_local_ca_mask(
    q,
    kv,
    window_size,
    wrap=True,
    flipped=False,
    query_phi=None,
    key_phi=None,
    shift_absolute: int | None = None,
    shift_fractional: float | None = None,
    shift_queries: int | None = None,
):
    n_objects = q.shape[1]
    n_inputs = kv.shape[1]
    device = q.device
    use_phi = query_phi is not None and key_phi is not None
    stride = n_inputs / n_objects

    # Validate shift parameters
    shift_count = sum(x is not None for x in [shift_absolute, shift_fractional, shift_queries])
    if shift_count > 0 and flipped:
        raise ValueError("LCA shift is not supported when flipped=True")

    if use_phi:
        batch_size = q.shape[0]
        if query_phi.dim() == 1:
            query_phi = query_phi.unsqueeze(0)
        if key_phi.dim() == 1:
            key_phi = key_phi.unsqueeze(0)
        assert query_phi.shape[0] == batch_size == key_phi.shape[0], "phi tensors must match batch dimension"
        if batch_size != 1:
            raise ValueError("Phi-based local CA mask currently supports batch size 1")

        q_phi = query_phi[0]
        k_phi = key_phi[0]
        assert q_phi.shape[-1] == n_objects, "query_phi must align with number of queries"
        assert k_phi.shape[-1] == n_inputs, "key_phi must align with number of keys"

        # For phi-based masks, apply shift as an angular offset to query_phi
        if shift_absolute is not None:
            # Convert absolute shift to angular offset: shift_absolute / n_inputs * 2*pi
            angular_offset = (shift_absolute / n_inputs) * 2 * torch.pi
            q_phi = q_phi + angular_offset
        elif shift_fractional is not None:
            # shift_fractional is already a fraction, convert to angular offset
            angular_offset = shift_fractional * 2 * torch.pi
            q_phi = q_phi + angular_offset
        elif shift_queries is not None:
            # Convert query positions to angular offset: shift_queries / n_objects * 2*pi
            angular_offset = (shift_queries / n_objects) * 2 * torch.pi
            q_phi = q_phi + angular_offset

        distances = _circular_distance(q_phi, k_phi, wrap)
        centers = torch.argmin(distances, dim=-1)
        if flipped:
            centers = torch.flip(centers, dims=(0,))
        mask = _window_mask_from_centers(
            centers,
            n_inputs,
            window_size,
            device,
            wrap,
            # Note: shift is already applied via angular offset above for phi-based masks
            shift_absolute=None,
            shift_fractional=None,
            shift_queries=None,
            stride=stride,
        )
    else:
        lca_func = get_local_ca_mask_flipped if flipped else get_local_ca_mask
        mask = lca_func(
            n_objects,
            n_inputs,
            window_size,
            stride,
            device,
            wrap,
            shift_absolute=shift_absolute,
            shift_fractional=shift_fractional,
            shift_queries=shift_queries,
        )

    assert not (~mask.any(dim=0)).any(), "Some columns are all False, increase window size"
    return mask.unsqueeze(0)
