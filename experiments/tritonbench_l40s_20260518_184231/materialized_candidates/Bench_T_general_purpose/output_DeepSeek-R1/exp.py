import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def exp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def exp(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    input = input.contiguous()
    
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
        if not out.is_contiguous():
            raise ValueError("out tensor must be contiguous")
    
    if input.device != out.device:
        raise ValueError("input and output must be on the same device")
    
    if not input.is_floating_point():
        raise TypeError("input must be a floating-point tensor")
    if not out.is_floating_point():
        raise TypeError("output must be a floating-point tensor")
    
    n_elements = input.numel()
    if n_elements == 0:
        return out
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
