import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def bessel_j1_kernel(x, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x_val = tl.load(x + offset, mask=mask)
    y_val = libdevice.bessel_j1(x_val)
    tl.store(out + offset, y_val, mask=mask)

def bessel_j1(input, *, out=None):
    N = input.numel()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.device == input.device

    block_size = 1024
    grid_size = triton.cdiv(N, block_size)
    grid = (grid_size, )

    bessel_j1_kernel[grid](
        input,
        out,
        N,
        block_size
    )

    return out
