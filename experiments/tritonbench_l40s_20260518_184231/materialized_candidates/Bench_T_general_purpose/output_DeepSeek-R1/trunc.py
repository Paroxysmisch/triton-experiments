import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def trunc_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.trunc(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def trunc(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Handle integer types by returning a copy
    if input.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        if out is None:
            return input.clone()
        else:
            out.copy_(input)
            return out
    
    # Handle floating-point types
    if not input.is_cuda:
        return torch.trunc(input, out=out)
    
    n_elements = input.numel()
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
        assert output.shape == input.shape, "out tensor must have the same shape as input"
        assert output.dtype == input.dtype, "out tensor must have the same dtype as input"
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    trunc_kernel[grid](input.data_ptr(), output.data_ptr(), n_elements, BLOCK_SIZE=1024)
    
    return output
