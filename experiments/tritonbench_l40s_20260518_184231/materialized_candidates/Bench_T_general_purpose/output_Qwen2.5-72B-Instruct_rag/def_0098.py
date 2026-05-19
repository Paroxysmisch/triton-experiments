import torch
import triton
import triton.language as tl
from triton.language.math import erf, tanh, sqrt, pow

# Triton kernel for exact GELU
@triton.jit
def gelu_none_kernel(input, other, alpha, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input.shape[0]
    
    input_vec = tl.load(input + offsets, mask=mask)
    other_vec = tl.load(other + offsets, mask=mask)
    
    # Compute the scaled subtraction
    sub_result = input_vec - alpha * other_vec
    
    # Compute the GELU function using the error function approximation
    gelu_result = 0.5 * sub_result * (1 + erf(sub_result * 0.7071067811))
    
    # Store the result
    tl.store(out + offsets, gelu_result, mask=mask)

# Triton kernel for approximate GELU using tanh
@triton.jit
def gelu_tanh_kernel(input, other, alpha, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input.shape[0]
    
    input_vec = tl.load(input + offsets, mask=mask)
    other_vec = tl.load(other + offsets, mask=mask)
    
    # Compute the scaled subtraction
    sub_result = input_vec - alpha * other_vec
    
    # Compute the GELU function using the tanh approximation
    gelu_result = 0.5 * sub_result * (1 + tanh(sqrt(2 / 3.141592653589793) * (sub_result + 0.044715 * pow(sub_result, 3))))
    
    # Store the result
    tl.store(out + offsets, gelu_result, mask=mask)

# Wrapper function
def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    if approximate == 'none':
        grid = (input.numel() + 1024 - 1) // 1024
        gelu_none_kernel[grid, 1024](input, other, alpha, out, BLOCK_SIZE=1024)
    elif approximate == 'tanh':
        grid = (input.numel() + 1024 - 1) // 1024
        gelu_tanh_kernel[grid, 1024](input, other, alpha, out, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    return out
