import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def bessel_j1_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    output_values = tl.math.j1(input_values)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def bessel_j1(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not input.is_cuda:
        raise RuntimeError("Input tensor must be on a CUDA device.")
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise RuntimeError("Output tensor must have the same shape as input tensor.")
        if out.device != input.device:
            raise RuntimeError("Output tensor must be on the same device as input tensor.")
    
    # Check if output is contiguous, otherwise use a temporary buffer
    if out.is_contiguous():
        out_buffer = out
    else:
        out_buffer = torch.empty_like(input).contiguous()
    
    input_flat = input.contiguous().view(-1)
    out_flat = out_buffer.view(-1)
    n_elements = input_flat.numel()
    
    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        bessel_j1_kernel[grid](input_flat, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Copy back if necessary
    if not out.is_contiguous():
        out.copy_(out_buffer)
    
    return out
