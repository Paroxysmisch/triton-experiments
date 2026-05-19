import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.abs(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def abs(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    input_flat = input.view(-1)
    N = input_flat.numel()
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.size() != input.size():
            raise ValueError("out tensor must have the same shape as input")
    out_flat = out.view(-1)
    if input_flat.device.type != 'cuda':
        raise ValueError("input must be a CUDA tensor")
    if input_flat.dtype not in (torch.float32, torch.float16, torch.int32, torch.int16, torch.int8):
        raise ValueError("Unsupported data type for input tensor")
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    abs_kernel[grid](input_flat, out_flat, N, BLOCK_SIZE=1024)
    return out
