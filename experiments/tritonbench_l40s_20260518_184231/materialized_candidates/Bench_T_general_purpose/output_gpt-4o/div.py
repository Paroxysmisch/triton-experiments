import triton
import triton.language as tl

@triton.jit
def div_kernel(
    input_ptr, other_ptr, out_ptr, 
    n_elements, 
    rounding_mode, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and other elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=1.0)  # Avoid division by zero

    # Perform division
    result = input_vals / other_vals

    # Apply rounding if specified
    if rounding_mode == 0:  # No rounding
        pass
    elif rounding_mode == 1:  # Floor
        result = tl.floor(result)
    elif rounding_mode == 2:  # Ceil
        result = tl.ceil(result)
    elif rounding_mode == 3:  # Truncate
        result = tl.trunc(result)

    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

import torch

def div(input, other, *, rounding_mode=None, out=None):
    # Promote inputs to the default scalar type (usually float32)
    input = input.to(torch.get_default_dtype())
    if isinstance(other, torch.Tensor):
        other = other.to(torch.get_default_dtype())
    else:
        other = torch.tensor(other, dtype=torch.get_default_dtype())

    # Determine the output shape based on broadcasting
    broadcast_shape = torch.broadcast_shapes(input.shape, other.shape)

    # Prepare the output tensor
    if out is None:
        out = torch.empty(broadcast_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == broadcast_shape, "Output tensor has incorrect shape"

    # Flatten tensors for processing
    input_flat = input.expand(broadcast_shape).contiguous().flatten()
    other_flat = other.expand(broadcast_shape).contiguous().flatten()
    out_flat = out.flatten()

    # Determine rounding mode for Triton kernel
    rounding_modes = {
        None: 0,  # No rounding
        'floor': 1,
        'ceil': 2,
        'trunc': 3
    }
    triton_rounding_mode = rounding_modes.get(rounding_mode, 0)

    # Launch the Triton kernel
    n_elements = out_flat.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    div_kernel[grid](
        input_flat, other_flat, out_flat, 
        n_elements, 
        triton_rounding_mode, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(broadcast_shape)
