import triton
import triton.language as tl

@triton.jit
def bessel_j1_kernel(X, Y, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(X + offsets, mask=mask)
    y = tl.math.bessel_j1(x)
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['size']
)
@triton.jit
def bessel_j1_kernel(X, Y, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(X + offsets, mask=mask)
    y = tl.math.bessel_j1(x)
    tl.store(Y + offsets, y, mask=mask)

def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    size = input.numel()
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    bessel_j1_kernel[grid](input, out, size, BLOCK_SIZE=128)
    return out
