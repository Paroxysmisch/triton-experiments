import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(
    inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and mask values
    inp_vals = tl.load(inp_ptr + offsets, mask=mask)
    mask_vals = tl.load(select_mask_ptr + offsets, mask=mask).to(tl.int1)

    # Calculate output positions using prefix sum
    prefix_sum_vals = tl.load(prefix_sum_ptr + offsets, mask=mask)
    out_offsets = tl.where(mask_vals, prefix_sum_vals, -1)

    # Store selected input values in the output buffer
    tl.store(out_ptr + out_offsets, inp_vals, mask=mask_vals)

import torch
import triton
import triton.language as tl

def masked_select(inp, select_mask):
    # Ensure input and mask are broadcastable
    if not broadcastable(inp.shape, select_mask.shape):
        raise ValueError("Input and mask must be broadcastable")

    # Flatten the mask
    select_mask = select_mask.flatten()

    # Compute prefix sum for determining output positions
    prefix_sum = torch.cumsum(select_mask, dim=0)

    # Flatten the input
    inp = inp.flatten()

    # Determine the number of elements and the number of selected elements
    n_elements = inp.numel()
    n_selected = select_mask.sum().item()

    # Allocate output buffer
    out = torch.empty(n_selected, dtype=inp.dtype, device=inp.device)

    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    masked_select_kernel[grid_size, BLOCK_SIZE](
        inp, select_mask, prefix_sum, out, n_elements, BLOCK_SIZE
    )

    return out

def broadcastable(shape1, shape2):
    try:
        for a, b in zip(shape1[::-1], shape2[::-1]):
            if a == 1 or b == 1 or a == b:
                continue
            return False
        return True
    except Exception:
        return False
