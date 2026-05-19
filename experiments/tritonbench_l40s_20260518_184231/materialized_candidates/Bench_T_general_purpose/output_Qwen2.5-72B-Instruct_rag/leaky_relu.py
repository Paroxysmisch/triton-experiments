import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(X, Y, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)
    
    # Compute Leaky ReLU
    y = tl.where(x >= 0, x, negative_slope * x)
    
    # Store the result
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def leaky_relu(input, negative_slope=0.01, inplace=False):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    # Ensure the tensor is on the same device
    device = input.device
    dtype = input.dtype
    
    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input, device=device, dtype=dtype)
    
    # Define grid and block sizes
    grid = (triton.cdiv(input.numel(), 1024),)
    block_size = 1024
    
    # Launch the kernel
    leaky_relu_kernel[grid](input, output, negative_slope, BLOCK_SIZE=block_size)
    
    return output
