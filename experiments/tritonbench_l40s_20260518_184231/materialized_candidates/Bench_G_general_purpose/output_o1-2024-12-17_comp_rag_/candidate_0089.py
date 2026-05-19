import torch
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # Load int8 data (representing float8) from X
    x_i8 = tl.load(X + offs, mask=mask, other=0)
    # Convert (reinterpret) int8 -> float16
    x_f16 = tl.cast(x_i8, tl.float16)
    # Store result in Y
    tl.store(Y + offs, x_f16, mask=mask)

def f8_to_f16(x):
    assert x.dtype == torch.int8, "Input must be of type torch.int8."
    assert x.is_cuda, "Input must be on CUDA device."
    out = torch.empty_like(x, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    kernel_f8_to_f16[grid](out, x, x.numel(), BLOCK_SIZE=1024)
    return out

@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # Load float16/float32 data from X
    x_val = tl.load(X + offs, mask=mask, other=0.0)
    # Convert float16/float32 -> int8 (representing float8)
    x_i8 = tl.cast(x_val, tl.int8)
    tl.store(Y + offs, x_i8, mask=mask)

def f16_to_f8(x):
    assert x.dtype in (torch.float16, torch.float32), "Input must be float16 or float32."
    assert x.is_cuda, "Input must be on CUDA device."
    out = torch.empty_like(x, dtype=torch.int8)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    kernel_f16_to_f8[grid](out, x, x.numel(), BLOCK_SIZE=1024)
    return out
