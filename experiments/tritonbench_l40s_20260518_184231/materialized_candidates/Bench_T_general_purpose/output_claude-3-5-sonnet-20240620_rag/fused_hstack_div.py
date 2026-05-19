import torch
import triton
import triton.language as tl

@triton.jit
def _fused_hstack_div(
    X_ptrs,
    X_dims,
    X_offsets,
    num_tensors,
    divisor_ptr,
    divisor_scalar,
    out_ptr,
    total_elements,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Calculate the correct tensor and offset within the stacked tensor
    tensor_index = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    offset_in_tensor = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(num_tensors):
        current_tensor_elements = X_dims[i]
        tensor_mask = offsets >= X_offsets[i]
        tensor_index = tl.where(tensor_mask, i, tensor_index)
        offset_in_tensor = tl.where(tensor_mask, offsets - X_offsets[i], offset_in_tensor)

    # Load elements from the appropriate tensors based on calculated indices
    stacked_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(num_tensors):
        tensor_mask = tensor_index == i
        tensor_ptr = X_ptrs[i]
        stacked_vals = tl.where(tensor_mask, tl.load(tensor_ptr + offset_in_tensor, mask=tensor_mask), stacked_vals)

    # Handle divisor: either scalar or tensor
    if divisor_scalar:
        divisor = divisor_scalar
    else:
        divisor = tl.load(divisor_ptr + offsets, mask=mask)

    if rounding_mode == "trunc":
        result = tl.trunc(stacked_vals / divisor)
    elif rounding_mode == "floor":
        result = tl.floor(stacked_vals / divisor)
    else:
        result = stacked_vals / divisor

    tl.store(out_ptr + offsets, result, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("tensors must be a sequence of tensors.")

    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("All elements in tensors must be torch.Tensor objects.")

    if not all(t.dtype == tensors[0].dtype for t in tensors):
        raise TypeError("All tensors must have the same dtype.")

    stacked_shape = list(tensors[0].shape)
    stacked_shape[0] = sum([t.shape[0] for t in tensors])
    total_elements = sum(t.numel() for t in tensors)

    if out is None:
        out = torch.empty(stacked_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    else:
        assert out.shape == tuple(stacked_shape)

    X_ptrs = [t.data_ptr() for t in tensors]
    X_dims = [t.shape[0] for t in tensors]
    X_offsets = [sum(X_dims[:i]) for i in range(len(X_dims))]
    num_tensors = len(tensors)

    divisor_scalar = False
    if isinstance(divisor, (int, float)):
        divisor_scalar = True
        divisor_ptr = None  # Dummy value, not used
    else:
        if not isinstance(divisor, torch.Tensor):
            raise TypeError("divisor must be a Tensor or a Number.")
        divisor = divisor.to(tensors[0].device)  # Ensure divisor is on the same device
        divisor_ptr = divisor.data_ptr()

    grid = lambda META: (triton.cdiv(total_elements, META['BLOCK_SIZE']),)
    _fused_hstack_div[grid](
        X_ptrs=X_ptrs,
        X_dims=X_dims,
        X_offsets=X_offsets,
        num_tensors=num_tensors,
        divisor_ptr=divisor_ptr,
        divisor_scalar=divisor_scalar,
        out_ptr=out.data_ptr(),
        total_elements=total_elements,
        rounding_mode=rounding_mode,
        BLOCK_SIZE=1024,  # Example block size, adjust as needed
    )
    return out
