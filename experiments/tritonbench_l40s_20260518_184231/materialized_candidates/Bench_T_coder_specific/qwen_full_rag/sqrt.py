import triton
import triton.language as tl
import torch

@triton.jit
def sqrt_kernel(x, y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x_val = tl.load(x + offset, mask=mask)
    y_val = tl.sqrt(x_val)
    tl.store(y + offset, y_val, mask=mask)

def sqrt(x, out=None):
    x = x.contiguous()
    if out is None:
        out = torch.empty_like(x)
    else:
        assert x.stride() == out.stride()
    assert out.is_contiguous()
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    sqrt_kernel[grid](x, out, N, BLOCK_SIZE=1024)
    return out
