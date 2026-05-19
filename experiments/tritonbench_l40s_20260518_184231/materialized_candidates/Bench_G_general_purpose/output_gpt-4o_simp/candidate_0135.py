import triton
import triton.language as tl

@triton.jit
def kernel_function(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting index for this block
    start_idx = block_idx * BLOCK_SIZE
    
    # Create a range of indices for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data from global memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the sine of each element
    y = tl.sin(x)
    
    # Store the result back to global memory
    tl.store(y_ptr + offsets, y, mask=mask)

import torch

def call_kernel(x):
    # Ensure input is a Torch tensor on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"
    
    # Allocate output tensor on the GPU
    y = torch.empty_like(x)
    
    # Define the block size (you can adjust this for performance tuning)
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed
    n_elements = x.numel()
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    kernel_function[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return y
