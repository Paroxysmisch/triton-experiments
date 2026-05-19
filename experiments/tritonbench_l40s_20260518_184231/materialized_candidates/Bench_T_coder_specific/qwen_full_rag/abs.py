import torch
import triton
import triton.language as tl

@triton.jit
def abs_kernel(X, Y, N, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    y = tl.abs(x)
    tl.store(Y + offset, y, mask=mask)

def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    assert out.is_contiguous()
    N = input.numel()
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    abs_kernel[grid](input, out, N, BLOCK_SIZE=1024)
    return out
