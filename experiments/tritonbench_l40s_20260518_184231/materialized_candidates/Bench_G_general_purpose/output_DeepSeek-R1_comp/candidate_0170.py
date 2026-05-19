import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_func(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_values = tl.load(a_ptr + offsets, mask=mask)
    b_values = tl.cos(a_values)
    tl.store(b_ptr + offsets, b_values, mask=mask)

def cos(a: torch.Tensor):
    assert a.is_cuda and a.is_contiguous(), "Input tensor must be a contiguous CUDA tensor."
    b = torch.empty_like(a)
    n_elements = a.numel()
    if n_elements == 0:
        return b
    sqrt_n = math.sqrt(n_elements)
    block_size = 1
    while block_size < sqrt_n:
        block_size *= 2
    grid_size = (n_elements + block_size - 1) // block_size
    cos_func[(grid_size,)](a, b, n_elements, BLOCK_SIZE=block_size)
    return b
