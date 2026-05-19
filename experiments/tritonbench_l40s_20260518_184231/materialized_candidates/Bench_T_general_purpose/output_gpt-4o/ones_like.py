import triton
import triton.language as tl

@triton.jit
def fill_ones_kernel(
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index
    block_idx = tl.program_id(0)
    # Calculate the start index for this block
    start_idx = block_idx * BLOCK_SIZE
    # Create a range of indices for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    # Fill the output with ones
    tl.store(output_ptr + offsets, 1.0, mask=mask)

import torch

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Determine the output tensor's properties
    dtype = dtype or input.dtype
    layout = layout or input.layout
    device = device or input.device
    memory_format = memory_format or torch.preserve_format
    
    # Create an empty tensor with the same size as the input
    output = torch.empty_like(input, dtype=dtype, layout=layout, device=device, memory_format=memory_format)
    
    # Determine the number of elements
    n_elements = output.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_ones_kernel[grid](output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Set requires_grad if necessary
    output.requires_grad_(requires_grad)
    
    return output
