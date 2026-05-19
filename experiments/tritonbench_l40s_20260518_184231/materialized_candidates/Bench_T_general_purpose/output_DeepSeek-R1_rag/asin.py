import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)

def asin(input, *, out=None):
    if not input.is_cuda:
        raise TypeError("Input tensor must be on CUDA device.")
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape does not match input tensor.")
        if not out.is_cuda:
            raise TypeError("Output tensor must be on CUDA device.")
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    asin_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
