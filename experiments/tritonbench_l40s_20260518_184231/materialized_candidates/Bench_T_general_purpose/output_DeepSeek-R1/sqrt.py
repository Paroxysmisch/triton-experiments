import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.sqrt(input_vals)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def sqrt(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not input.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")
    
    if out is None:
        if input.dtype.is_floating_point:
            dtype = input.dtype
        else:
            dtype = torch.float32
        out = torch.empty_like(input, dtype=dtype)
    else:
        if not out.is_cuda:
            raise RuntimeError("Output tensor must be on CUDA device")
        if not out.dtype.is_floating_point:
            raise RuntimeError("Output tensor must be a floating-point type")
        if out.shape != input.shape:
            raise RuntimeError("Output tensor shape must match input tensor shape")
        dtype = out.dtype
    
    input = input.to(dtype)
    num_elements = input.numel()
    
    if num_elements == 0:
        return out
    
    block_size = 1024
    grid_size = (num_elements + block_size - 1) // block_size
    
    sqrt_kernel[grid_size](
        input,
        out,
        num_elements,
        BLOCK_SIZE=block_size
    )
    
    return out
