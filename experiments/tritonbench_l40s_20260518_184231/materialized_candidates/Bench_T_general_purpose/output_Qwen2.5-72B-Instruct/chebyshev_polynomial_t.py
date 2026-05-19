import triton
import triton.language as tl

@triton.jit
def chebyshev_polynomial_kernel(input_ptr, n_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask)

    result = tl.where(n == 0, 1.0, input)
    result = tl.where(n == 1, input, result)

    for i in range(2, 6):
        t_i_minus_1 = tl.where(n == i - 1, input, result)
        t_i_minus_2 = tl.where(n == i - 2, 1.0, t_i_minus_1)
        result = tl.where(n == i, 2 * input * t_i_minus_1 - t_i_minus_2, result)

    result = tl.where((n >= 6) & (tl.abs(input) <= 1.0), tl.cos(n * tl.acos(input)), result)

    tl.store(out_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def chebyshev_polynomial_kernel(input_ptr, n_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask)

    result = tl.where(n == 0, 1.0, input)
    result = tl.where(n == 1, input, result)

    for i in range(2, 6):
        t_i_minus_1 = tl.where(n == i - 1, input, result)
        t_i_minus_2 = tl.where(n == i - 2, 1.0, t_i_minus_1)
        result = tl.where(n == i, 2 * input * t_i_minus_1 - t_i_minus_2, result)

    result = tl.where((n >= 6) & (tl.abs(input) <= 1.0), tl.cos(n * tl.acos(input)), result)

    tl.store(out_ptr + offsets, result, mask=mask)

def chebyshev_polynomial_t(input, n, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 128),)

    chebyshev_polynomial_kernel[grid](
        input, n, out, n_elements, BLOCK_SIZE=128
    )

    return out
