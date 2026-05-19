import torch
import triton
import triton.language as tl

# Out-of-place kernel
@triton.jit
def mul2_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(y_ptr + offsets, y, mask=mask)

# In-place kernel
@triton.jit
def mul2_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(x_ptr + offsets, y, mask=mask)

# Wrapper for out-of-place operation
def triton_mul2(x: torch.Tensor, BLOCK_SIZE: int = 1024):
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be contiguous and on CUDA"
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_kernel[grid](x.data_ptr(), y.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

# Wrapper for in-place operation
def triton_mul2_inplace(x: torch.Tensor, BLOCK_SIZE: int = 1024):
    assert x.is_cuda and x.is_contiguous(), "Input tensor must be contiguous and on CUDA"
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul2_inplace_kernel[grid](x.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x
