import logging
import torch
import triton
import triton.language as tl


@triton.jit
def signbit_kernel(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.where(x_values < 0, 1, 0)
    tl.store(y + offsets, y_values, mask=mask)


def triton_signbit(input, out=None):
    logging.debug("GEMS SIGNBIT")
    input = input.contiguous()
    if out == None:
        out = torch.zeros_like(input)
    else:
        out = out.contiguous()
    assert input.is_cuda and out.is_cuda
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
