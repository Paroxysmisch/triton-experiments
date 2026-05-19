import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Compute the block start and end indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't read out of bounds
    mask = offsets < n_elements
    
    # Load the input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute log and then tanh
    log_data = tl.log(input_data)
    tanh_data = tl.tanh(log_data)
    
    # Store the result
    tl.store(output_ptr + offsets, tanh_data, mask=mask)

import torch

def log_tanh(input, out=None):
    # Check if all elements are positive
    if not torch.all(input > 0):
        raise ValueError("All elements of the input tensor must be positive for the log function.")
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Number of elements in the input tensor
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024  # This can be tuned based on hardware and input size
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
