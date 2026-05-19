import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(x, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x.shape[0]
    x_val = tl.load(x + offsets, mask=mask)
    y_val = 1 / tl.sqrt(x_val)
    tl.store(Y + offsets, y_val, mask=mask)

def rsqrt(x, *, out=None):
    if out is None:
        out = torch.empty_like(x)
    assert x.is_contiguous()
    assert out.is_contiguous()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    rsqrt_kernel[grid](x, out, BLOCK_SIZE=BLOCK_SIZE)
    return out
