import triton
import triton.language as tl

@triton.jit
def log_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the index of the current element
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute natural logarithm
    result = tl.log(input_data)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

import torch

def log(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Example block size, can be tuned
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    log_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
