import triton
import triton.language as tl

@triton.jit
def logit_kernel(input_ptr, output_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    # Get the program index
    pid = tl.program_id(0)
    
    # Create a range of offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    x = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Apply clamping based on eps
    if eps is not None:
        x = tl.where(x < eps, eps, x)
        x = tl.where(x > 1 - eps, 1 - eps, x)
    
    # Compute logit
    logit_value = tl.log(x / (1 - x))
    
    # Handle NaN for out of bound values when eps is None
    if eps is None:
        logit_value = tl.where((x < 0) | (x > 1), float('nan'), logit_value)
    
    # Store result
    tl.store(output_ptr + offsets, logit_value, mask=offsets < n_elements)


import torch

def logit(input, eps=None, *, out=None):
    # Prepare input tensor
    input_tensor = input.contiguous()
    n_elements = input_tensor.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input_tensor)
    
    # Determine block size
    BLOCK_SIZE = 1024  # You can adjust this based on your hardware capabilities
    
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    logit_kernel[grid](input_tensor, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
