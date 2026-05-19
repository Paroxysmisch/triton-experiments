import torch
import triton
import triton.language as tl

def f8_to_f16(x):
    @triton.jit
    def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    assert x.type() == torch.int8 and x.is_cuda
    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel_f8_to_f16[grid](ret, triton.reinterpret(x, tl.float8e5), ret.numel(), BLOCK_SIZE=1024)
    return ret


def f16_to_f8(x):
    @triton.jit
    def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    assert any(x.type() == torch_type for torch_type in (torch.float16, torch.float32)) and x.is_cuda
    ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel_f16_to_f8[grid](ret, triton.reinterpret(x, tl.float8e5), ret.numel(), BLOCK_SIZE=1024)
    return ret
