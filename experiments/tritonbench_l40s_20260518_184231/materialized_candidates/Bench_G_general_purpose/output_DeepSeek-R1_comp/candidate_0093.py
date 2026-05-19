import torch
import triton
import triton.language as tl

@triton.jit
def kernel_f8_to_f16(
    X_ptr,
    Y_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(X_ptr + offsets, mask=mask, dtype=tl.float8)
    y = x.to(tl.float16)
    tl.store(Y_ptr + offsets, y, mask=mask)

def f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8 and x.is_cuda, "Input must be a CUDA int8 tensor"
    y = torch.empty_like(x, dtype=torch.float16)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_f8_to_f16[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y

@triton.jit
def kernel_f16_to_f8(
    X_ptr,
    Y_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,  # tl.float16 or tl.float32
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(X_ptr + offsets, mask=mask, dtype=INPUT_DTYPE)
    x_f8 = x.to(tl.float8)
    tl.store(Y_ptr + offsets, x_f8.to(tl.int8), mask=mask)

def f16_to_f8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype in (torch.float16, torch.float32) and x.is_cuda, "Input must be CUDA float16/float32 tensor"
    y = torch.empty_like(x, dtype=torch.int8)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if x.dtype == torch.float16:
        kernel_f16_to_f8[grid](x, y, n_elements, INPUT_DTYPE=tl.float16, BLOCK_SIZE=1024)
    else:
        kernel_f16_to_f8[grid](x, y, n_elements, INPUT_DTYPE=tl.float32, BLOCK_SIZE=1024)
    return y
