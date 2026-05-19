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
        y = x.to(tl.float16)
        tl.store(Y + offs, y, mask=mask)

    assert x.dtype == torch.int8, "Input tensor must be of type torch.int8"
    assert x.device.type == 'cuda', "Input tensor must be on a CUDA device"

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel[grid](ret, x, x.numel(), BLOCK_SIZE=1024)
    return ret

import torch
import triton
import triton.language as tl

def f16_to_f8(x):
    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = x.to(tl.int8)
        tl.store(Y + offs, y, mask=mask)

    assert x.dtype in [torch.float16, torch.float32], "Input tensor must be of type torch.float16 or torch.float32"
    assert x.device.type == 'cuda', "Input tensor must be on a CUDA device"

    ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel[grid](ret, x, x.numel(), BLOCK_SIZE=1024)
    return ret
