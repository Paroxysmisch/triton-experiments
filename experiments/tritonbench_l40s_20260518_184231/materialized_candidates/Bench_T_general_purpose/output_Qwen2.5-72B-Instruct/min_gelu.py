import triton
import triton.language as tl

@triton.jit
def gelu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr, APPROXIMATE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    if APPROXIMATE:
        x = 0.5 * x * (1 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    else:
        x = x * tl.cdf(x)
    
    tl.store(Y + offsets, x, mask=mask)

@triton.jit
def min_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr, DIM: tl.constexpr, KEEP_DIM: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    min_val = tl.min(x, axis=DIM, keepdim=KEEP_DIM)
    tl.store(Y + offsets, min_val, mask=mask)

import torch
import triton
import triton.language as tl

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Determine the approximate method
    approximate = approximate.lower()
    if approximate not in ['none', 'tanh']:
        raise ValueError("Approximate method must be 'none' or 'tanh'")
    
    # Compute GELU
    gelu_output = torch.empty_like(input)
    grid = (triton.cdiv(input.numel(), 1024),)
    gelu_kernel[grid](input, gelu_output, input.numel(), BLOCK_SIZE=1024, APPROXIMATE=(approximate == 'tanh'))
    
    # Compute the minimum value
    if dim is None:
        min_val = torch.min(gelu_output)
        if out is not None:
            out.copy_(min_val)
        return min_val
    else:
        min_val, min_indices = torch.min(gelu_output, dim=dim, keepdim=keepdim)
        if out is not None:
            out.copy_(min_val)
        return min_val, min_indices

# Example usage
input_tensor = torch.randn(4, 5, 6)
result = min_gelu(input_tensor, dim=1, keepdim=True, approximate='tanh')
print(result)
