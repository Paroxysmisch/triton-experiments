import triton
import triton.language as tl
from typing import List

def broadcastable(shape_a, shape_b):
    """
    Checks if two shapes are broadcastable under standard broadcasting rules.
    """
    for a, b in zip(shape_a[::-1], shape_b[::-1]):
        if not (a == b or a == 1 or b == 1):
            return False
    return True

@triton.autotune(
    configs=[
        triton.Config(meta={'BLOCK_SIZE': 64, 'num_warps': 1}, num_stages=2),
        triton.Config(meta={'BLOCK_SIZE': 128, 'num_warps': 2}, num_stages=2),
        triton.Config(meta={'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def masked_select_kernel(
    inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr,
    n_elements,  
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    sel_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0)
    sel_mask_bool = sel_mask.to(tl.int1)

    out_pos = tl.load(prefix_sum_ptr + offsets, mask=mask & sel_mask_bool, other=0)
    tl.store(out_ptr + out_pos, inp, mask=mask & sel_mask_bool)

def masked_select(inp, select_mask):
    """
    Applies element-wise masked select using Triton.
    """
    import torch

    # Ensure compatibility via broadcasting
    if not broadcastable(inp.shape, select_mask.shape):
        raise ValueError("Input shapes are not broadcastable.")

    # Broadcast inputs
    b_inp, b_select_mask = torch.broadcast_tensors(inp, select_mask)

    # Flatten
    flat_inp = b_inp.flatten()
    flat_mask = b_select_mask.flatten()
    n_elements = flat_inp.numel()

    # Compute prefix sums to determine positions
    prefix_sum = torch.cumsum(flat_mask, dim=0) - 1
    total_selected = prefix_sum[-1].item() + 1 if n_elements > 0 else 0

    # Prepare output
    out = torch.empty((total_selected,), dtype=flat_inp.dtype, device=flat_inp.device)

    # Launch kernel
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    masked_select_kernel[grid](
        flat_inp,
        flat_mask,
        prefix_sum,
        out,
        n_elements,
    )
    return out
