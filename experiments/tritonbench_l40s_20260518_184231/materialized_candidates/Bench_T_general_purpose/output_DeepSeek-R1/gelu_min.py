import torch
from torch import Tensor
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    approximate: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    if approximate == "none":
        cdf = 0.5 * (1 + tl.erf(x * 0.7071067811865475))
        output = x * cdf
    else:
        a = 0.044715
        sqrt_2_over_pi = 0.7978845608
        x_cubed = x * x * x
        inner = sqrt_2_over_pi * (x + a * x_cubed)
        tanh_inner = tl.tanh(inner)
        output = 0.5 * x * (1 + tanh_inner)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu_triton(input: Tensor, approximate: str) -> Tensor:
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](input, output, approximate, n_elements, BLOCK_SIZE=1024)
    return output

def gelu_min(
    input: Tensor,
    approximate: str = 'none',
    dim: Optional[int] = None,
    keepdim: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor or Tuple[Tensor, Tensor]:
    gelu_output = gelu_triton(input, approximate)
    
    if dim is not None:
        result = torch.min(gelu_output, dim=dim, keepdim=keepdim)
        return result
    else:
        min_val = torch.amin(gelu_output, dim=dim, keepdim=keepdim)
        if out is not None:
            out.copy_(min_val)
            return out
        else:
            return min_val
