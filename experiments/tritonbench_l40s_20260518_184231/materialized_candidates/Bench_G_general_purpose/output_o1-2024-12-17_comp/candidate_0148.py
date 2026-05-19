import torch
import triton
import triton.language as tl

@triton.jit
def mul2_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = x * 2
    tl.store(output_ptr + offsets, x, mask=mask)

@triton.jit
def mul2_inplace_kernel(
    input_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = x * 2
    tl.store(input_ptr + offsets, x, mask=mask)

def triton_mul2(input_tensor):
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=1024)
    return output

def triton_mul2_inplace(input_tensor):
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](input_tensor, n_elements, BLOCK_SIZE=1024)
    return input_tensor
