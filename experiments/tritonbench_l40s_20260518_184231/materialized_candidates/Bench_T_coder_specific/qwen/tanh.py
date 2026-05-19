import torch
import triton
import triton.language as tl

# Load the Triton kernel
from .tanh_kernel import tanh_kernel

# Triton kernel configuration
BLOCK_SIZE = 128

@torch.jit.script
def tanh(input, out=None):
    # Determine the number of elements
    n_elements = input.numel()
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    tanh_kernel[grid_size, BLOCK_SIZE](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
