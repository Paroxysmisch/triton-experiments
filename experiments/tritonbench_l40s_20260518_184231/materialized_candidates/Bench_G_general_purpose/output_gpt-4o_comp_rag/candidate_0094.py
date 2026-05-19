import torch
import triton
import triton.language as tl

# Kernel to convert float8 data stored as int8 to float16
@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # Assuming reinterpretation of int8 to float8 is handled by triton.reinterpret
    x = tl.reinterpret(x, tl.float8)
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x):
    assert x.dtype == torch.int8, "Input tensor must be of type torch.int8"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel_f8_to_f16[grid](ret, x, x.numel(), BLOCK_SIZE=1024)
    return ret

# Kernel to convert float16 or float32 data to float8 stored as int8
@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    # Assuming reinterpretation of float16/float32 to float8 is handled by triton.reinterpret
    x = tl.reinterpret(x, tl.float8)
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x):
    assert x.dtype in [torch.float16, torch.float32], "Input tensor must be of type torch.float16 or torch.float32"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    ret = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    kernel_f16_to_f8[grid](ret, x, x.numel(), BLOCK_SIZE=1024)
    return ret
