import torch
import triton
import triton.language as tl

# Triton kernel for converting float8 to float16
@triton.jit
def kernel_f8_to_f16(X, Y, pid, BLOCK_SIZE: tl.constexpr):
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < X.shape[0]
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

# Function to invoke the kernel for float8 to float16 conversion
def f8_to_f16(x: torch.Tensor):
    assert x.dtype == torch.int8 and x.is_cuda
    y = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(x.numel(), BLOCK_SIZE),)
    kernel_f8_to_f16[grid](x, y, 0, BLOCK_SIZE)
    return y

# Triton kernel for converting float16 to float8
@triton.jit
def kernel_f16_to_f8(X, Y, pid, BLOCK_SIZE: tl.constexpr):
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < X.shape[0]
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

# Function to invoke the kernel for float16 to float8 conversion
def f16_to_f8(x: torch.Tensor):
    assert x.dtype == torch.float16 and x.is_cuda
    y = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(x.numel(), BLOCK_SIZE),)
    kernel_f16_to_f8[grid](x, y, 0, BLOCK_SIZE)
    return y
