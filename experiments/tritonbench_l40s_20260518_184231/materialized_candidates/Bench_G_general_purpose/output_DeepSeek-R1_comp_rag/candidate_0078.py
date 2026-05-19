import torch
import triton
import triton.language as tl

def f8_to_f16(x, dtype):
    @triton.jit
    def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)
    
    assert x.dtype == torch.int8, "Input must be torch.int8"
    assert x.is_cuda, "Input must be on CUDA device"
    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype_tl = getattr(tl, dtype)
    kernel_f8_to_f16[grid](ret, triton.reinterpret(x, dtype_tl), x.numel(), BLOCK_SIZE=1024)
    return ret

def f16_to_f8(x, dtype):
    @triton.jit
    def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr, DTYPE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x_val = tl.load(X + offs, mask=mask)
        x_f8 = tl.astype(x_val, DTYPE)
        x_i8 = tl.reinterpret(x_f8, tl.int8)
        tl.store(Y + offs, x_i8, mask=mask)
    
    assert x.dtype in [torch.float16, torch.float32], "Input must be float16 or float32"
    assert x.is_cuda, "Input must be on CUDA device"
    ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype_tl = getattr(tl, dtype)
    kernel_f16_to_f8[grid](ret, x, x.numel(), BLOCK_SIZE=1024, DTYPE=dtype_tl)
    return ret
