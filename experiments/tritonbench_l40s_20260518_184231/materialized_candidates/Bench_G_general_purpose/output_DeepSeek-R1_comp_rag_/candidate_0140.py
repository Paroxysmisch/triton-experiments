import torch
import triton
import triton.language as tl

# Out-of-place kernel
@triton.jit
def mul2_kernel(
    x_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(z_ptr + offsets, output, mask=mask)

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
    output = x * 2
    tl.store(x_ptr + offsets, output, mask=mask)

# Wrapper function for out-of-place operation
def triton_mul2(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Wrapper function for in-place operation
def triton_mul2_inplace(x: torch.Tensor):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    mul2_inplace_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
