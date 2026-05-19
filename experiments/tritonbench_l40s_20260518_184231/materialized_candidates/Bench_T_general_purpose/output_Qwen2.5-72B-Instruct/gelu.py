import triton
import triton.language as tl

@triton.jit
def gelu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    if approximate == 0:  # 'none'
        # Exact GELU: x * Φ(x)
        cdf = 0.5 * (1.0 + tl.math.erf(x / tl.sqrt(2.0)))
        y = x * cdf
    else:  # 'tanh'
        # Approximate GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        x3 = x * x * x
        inner = x + 0.044715 * x3
        inner_scaled = tl.sqrt(2.0 / tl.pi) * inner
        tanh_val = tl.math.tanh(inner_scaled)
        y = 0.5 * x * (1.0 + tanh_val)
    
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def gelu(input, approximate='none'):
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Convert input to a contiguous tensor
    input = input.contiguous()
    output = torch.empty_like(input)
    
    # Define grid and block sizes
    BLOCK_SIZE = 256
    grid = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    approximate_flag = 0 if approximate == 'none' else 1
    gelu_kernel[grid, BLOCK_SIZE](input, output, input.numel(), BLOCK_SIZE, approximate_flag)
    
    return output
