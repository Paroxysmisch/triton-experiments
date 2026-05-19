import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(inp_ptr, mask_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Check if offsets are within bounds
    mask = offsets < n_elements
    
    # Load input and mask values
    inp = tl.load(inp_ptr + offsets, mask=mask, other=0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Compute prefix sum of the mask to get output offsets
    prefix_sum = tl.cumsum(mask_vals, axis=0)
    output_offsets = prefix_sum - 1
    
    # Select elements where mask is true and store in output
    selected_elements = tl.where(mask_vals, inp, 0)
    tl.store(out_ptr + output_offsets, selected_elements, mask=mask_vals)

import torch

def masked_select(inp, mask):
    assert inp.shape == mask.shape, "Input and mask must have the same shape"
    
    # Flatten the input and mask tensors
    inp_flat = inp.flatten()
    mask_flat = mask.flatten()
    
    # Allocate output tensor with enough space
    num_selected = mask_flat.sum().item()
    out_flat = torch.empty(num_selected, dtype=inp.dtype, device=inp.device)
    
    # Define block size and grid size
    BLOCK_SIZE = 1024
    grid = (inp_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    masked_select_kernel[grid](
        inp_flat, mask_flat, out_flat, inp_flat.numel(), BLOCK_SIZE
    )
    
    return out_flat
