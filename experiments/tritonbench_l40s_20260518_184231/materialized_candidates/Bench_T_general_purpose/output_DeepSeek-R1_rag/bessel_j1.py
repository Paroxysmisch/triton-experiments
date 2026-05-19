import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def bessel_j1_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = tl.math.j1(input_vals)
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def bessel_j1(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA"
    if out is not None:
        assert out.is_cuda, "Output tensor must be on CUDA"
        assert out.shape == input.shape, "Output tensor shape mismatch"
    else:
        out = torch.empty_like(input)
    
    input_flat = input.view(-1)
    output_flat = out.view(-1)
    N = input_flat.numel()
    
    BLOCK_SIZE = 1024  # Commonly used block size for efficient GPU processing
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    
    bessel_j1_kernel[grid](input_ptr=input_flat, output_ptr=output_flat, N=N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
