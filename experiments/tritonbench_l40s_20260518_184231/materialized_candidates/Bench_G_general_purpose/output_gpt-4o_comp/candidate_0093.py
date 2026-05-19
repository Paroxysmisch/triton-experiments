import torch
import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 1024

@triton.jit
def kernel_f8_to_f16(X_ptr, Y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    # Load int8 data and reinterpret as float8
    x = tl.load(X_ptr + offs, mask=mask, other=0).to(tl.float8)
    
    # Store the result as float16
    tl.store(Y_ptr + offs, x.to(tl.float16), mask=mask)

def f8_to_f16(X):
    assert X.dtype == torch.int8, "Input tensor must be of type torch.int8"
    assert X.is_cuda, "Input tensor must be on a CUDA device"

    num_elements = X.numel()
    Y = torch.empty(num_elements, dtype=torch.float16, device=X.device)

    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    kernel_f8_to_f16[grid](X, Y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return Y

@triton.jit
def kernel_f16_to_f8(X_ptr, Y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements

    # Load float16 or float32 data
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)

    # Store the result as int8 (interpreted as float8)
    tl.store(Y_ptr + offs, x.to(tl.float8).to(tl.int8), mask=mask)

def f16_to_f8(X):
    assert X.dtype in [torch.float16, torch.float32], "Input tensor must be of type torch.float16 or torch.float32"
    assert X.is_cuda, "Input tensor must be on a CUDA device"

    num_elements = X.numel()
    Y = torch.empty(num_elements, dtype=torch.int8, device=X.device)

    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    kernel_f16_to_f8[grid](X, Y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    return Y
