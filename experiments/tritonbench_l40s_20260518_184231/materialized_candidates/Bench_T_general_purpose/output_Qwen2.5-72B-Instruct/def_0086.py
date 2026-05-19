import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(input_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute log and tanh
    log_vec = tl.log(input_vec)
    tanh_vec = tl.tanh(log_vec)
    
    tl.store(out_ptr + offsets, tanh_vec, mask=mask)

import torch
import triton
import triton.language as tl

def log_tanh(input, out=None):
    if not torch.is_tensor(input):
        raise TypeError("Input must be a Tensor")
    
    if torch.any(input <= 0):
        raise ValueError("All input elements must be positive for the logarithm function to be defined.")
    
    if out is None:
        out = torch.empty_like(input)
    else:
        if not torch.is_tensor(out):
            raise TypeError("Output must be a Tensor")
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    log_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
