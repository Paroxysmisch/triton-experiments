import torch
import triton
import triton.language as tl

def f8_to_f16(x):
    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.cast(x, tl.float16)
        tl.store(Y + offs, y, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel[grid](ret, triton.reinterpret(x, tl.int8), ret.numel(), BLOCK_SIZE=1024)
    return ret

def f16_to_f8(x, dtype='float8_e4m3'):
    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.cast(x, tl.int8)
        tl.store(Y + offs, y, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype = getattr(tl, dtype)
    kernel[grid](ret, x, ret.numel(), BLOCK_SIZE=1024)
    return ret
