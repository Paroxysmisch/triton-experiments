import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def i0_kernel(
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
    sum = tl.zeros(x.shape, dtype=x.dtype)
    term = tl.ones(x.shape, dtype=x.dtype)
    sum += term

    for k in range(1, 25):
        term = term * (x * x) / 4.0 / (k * k)
        sum += term

    tl.store(output_ptr + offsets, sum, mask=mask)

def i0(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if not input.is_floating_point():
        input = input.to(torch.float32)
    
    output_dtype = input.dtype
    input = input.contiguous()
    
    if out is not None:
        if not out.is_floating_point():
            raise TypeError("out tensor must be a floating point tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != output_dtype:
            raise TypeError(f"out tensor must have dtype {output_dtype}, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError("out tensor must be contiguous")
    else:
        out = torch.empty_like(input, dtype=output_dtype)
    
    n_elements = input.numel()
    if n_elements == 0:
        return out
    
    input_flat = input.view(-1)
    output_flat = out.view(-1)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    i0_kernel[grid](input_flat, output_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
