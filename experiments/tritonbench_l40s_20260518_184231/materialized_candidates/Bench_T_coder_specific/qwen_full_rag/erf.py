import torch
import triton
import triton.language as tl

@triton.jit
def erf_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    y = tl.erf(x)
    tl.store(Y + offset, y, mask=mask)

def erf(input, *, out=None):
    assert input.is_contiguous(), "Input must be contiguous"
    if out == None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
        assert out.shape == input.shape
    assert out.dtype == input.dtype
    assert out.device.type == input.device.type

    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    erf_kernel[grid](input, out, N, BLOCK_SIZE=1024)
    return out
