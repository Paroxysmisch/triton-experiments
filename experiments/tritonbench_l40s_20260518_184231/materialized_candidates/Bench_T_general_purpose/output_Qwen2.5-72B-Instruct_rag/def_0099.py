import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, pow, sqrt, mean

# GELU kernels
@triton.jit
def gelu_none_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = 0.5 * x_fp32 * (1 + erf(x_fp32 * 0.7071067811))
    tl.store(Y + offsets, y, mask=mask)

@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    y = 0.5 * x_fp32 * (1 + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32, 2))))
    tl.store(Y + offsets, y, mask=mask)

# Standard Deviation kernel
@triton.jit
def std_kernel(X, Y, N, correction, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x_mean = mean(x, axis=0)
    x_centered = x - x_mean
    x_squared = x_centered * x_centered
    x_var = mean(x_squared, axis=0)
    x_std = sqrt(x_var * (N / (N - correction)))
    tl.store(Y + pid, x_std, mask=pid < N)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Flatten the input tensor for simplicity
    input_flat = input.flatten()
    N = input_flat.numel()
    
    # Allocate output tensor for GELU result
    gelu_result = torch.empty_like(input_flat)
    
    # Apply GELU activation
    if approximate == 'none':
        grid = (N + 1024 - 1) // 1024
        gelu_none_kernel[grid, 1024](input_flat, gelu_result, N, BLOCK_SIZE=1024)
    elif approximate == 'tanh':
        grid = (N + 1024 - 1) // 1024
        gelu_tanh_kernel[grid, 1024](input_flat, gelu_result, N, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    # Reshape the GELU result to match the original input shape
    gelu_result = gelu_result.view_as(input)
    
    # Compute the standard deviation
    if dim is None:
        dim = tuple(range(input.dim()))
    if isinstance(dim, int):
        dim = (dim,)
    
    # Allocate output tensor for standard deviation
    std_result = torch.empty(input.size(), device=input.device, dtype=input.dtype)
    
    # Apply standard deviation computation
    for d in dim:
        grid = (input.size(d) + 1024 - 1) // 1024
        std_kernel[grid, 1024](gelu_result, std_result, input.size(d), correction, BLOCK_SIZE=1024)
    
    # Reduce dimensions if keepdim is False
    if not keepdim:
        std_result = std_result.squeeze(dim)
    
    # If an output tensor is provided, store the result in it
    if out is not None:
        out.copy_(std_result)
        return out
    else:
        return std_result

import torch

# Sample input
input = torch.randn(4, 4, 4)

# Test the function
result = gelu_std(input, dim=1, keepdim=True, correction=1, approximate='none')

# Print the result
print(result)
