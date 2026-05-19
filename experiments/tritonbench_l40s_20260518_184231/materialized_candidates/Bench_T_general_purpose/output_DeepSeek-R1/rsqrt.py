import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def rsqrt_kernel(
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
    output_values = 1.0 / tl.sqrt(input_values)
    tl.store(output_ptr + offsets, output_values, mask=mask)

def rsqrt(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA device"
    if not input.is_contiguous():
        input = input.contiguous()
    if out is not None:
        assert out.is_cuda, "Output tensor must be on CUDA device"
        assert out.shape == input.shape, "Output tensor shape must match input"
        if not out.is_contiguous():
            out = out.contiguous()
    else:
        out = torch.empty_like(input)
    n_elements = input.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rsqrt_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
