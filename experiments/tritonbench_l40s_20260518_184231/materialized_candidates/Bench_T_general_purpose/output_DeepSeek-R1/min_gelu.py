import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def gelu_exact_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_erf = tl.math.erf(x * 0.7071067811865476)  # 1/sqrt(2)
    gelu = x * 0.5 * (1.0 + x_erf)
    tl.store(output_ptr + offsets, gelu, mask=mask)

@triton.jit
def gelu_approx_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    approx = x + 0.044715 * x * x * x
    tanh_term = tl.math.tanh(sqrt_2_over_pi * approx)
    gelu = 0.5 * x * (1.0 + tanh_term)
    tl.store(output_ptr + offsets, gelu, mask=mask)

def min_gelu(input: Tensor, dim=None, keepdim=False, approximate='none', out=None) -> Tensor:
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Compute GELU using Triton kernel
    gelu_output = torch.empty_like(input)
    n_elements = gelu_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    if n_elements == 0:
        raise RuntimeError("Input tensor has no elements")
    
    if approximate == 'none':
        gelu_exact_kernel[grid](input, gelu_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        gelu_approx_kernel[grid](input, gelu_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute min
    if dim is not None:
        return torch.min(gelu_output, dim=dim, keepdim=keepdim)
    else:
        min_val = torch.min(gelu_output)
        if out is not None:
            out.copy_(min_val)
            return out
        else:
            return min_val
