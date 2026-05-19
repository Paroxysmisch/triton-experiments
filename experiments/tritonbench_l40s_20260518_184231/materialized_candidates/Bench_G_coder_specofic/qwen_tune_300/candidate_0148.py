import torch
import triton
import triton.language as tl

@triton.jit
def mul2_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    x = x * 2
    tl.store(output + offsets, x, mask=mask)

@triton.jit
def mul2_inplace_kernel(
    input,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    x = x * 2
    tl.store(input + offsets, x, mask=mask)

def triton_mul2(input):
    output = torch.empty_like(input)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    return output

def triton_mul2_inplace(input):
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](input, n_elements, BLOCK_SIZE=1024)
    return input
