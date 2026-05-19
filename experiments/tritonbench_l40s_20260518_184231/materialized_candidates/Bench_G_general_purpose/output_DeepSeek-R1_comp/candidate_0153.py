import torch
import triton
import triton.language as tl

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.sin(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def sin_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=4)
    return output
