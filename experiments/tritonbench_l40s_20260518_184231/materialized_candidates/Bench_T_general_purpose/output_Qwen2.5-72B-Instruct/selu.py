import triton
import triton.language as tl

@triton.jit
def selu_kernel(X, Y, alpha, scale, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    # Compute SELU
    pos = tl.where(x > 0, x, 0)
    neg = tl.where(x <= 0, alpha * (tl.exp(x) - 1), 0)
    y = scale * (pos + neg)
    
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def selu_kernel(X, Y, alpha, scale, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    # Compute SELU
    pos = tl.where(x > 0, x, 0)
    neg = tl.where(x <= 0, alpha * (tl.exp(x) - 1), 0)
    y = scale * (pos + neg)
    
    tl.store(Y + offsets, y, mask=mask)

def selu(input, inplace=False):
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    selu_kernel[grid](input, output, alpha, scale, N, BLOCK_SIZE=1024)
    
    return output

import torch

# Create a random input tensor
input_tensor = torch.randn(1024)

# Apply SELU
output_tensor = selu(input_tensor)

# Inplace application
selu(input_tensor, inplace=True)
