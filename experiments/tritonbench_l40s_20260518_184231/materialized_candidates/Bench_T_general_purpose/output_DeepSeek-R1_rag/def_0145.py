import triton
import triton.language as tl
import torch
import math

@triton.jit
def _polygamma_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    input_col_stride,
    output_row_stride,
    output_col_stride,
    n: tl.constexpr,
    sign: tl.constexpr,
    factorial_n: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    input_row = input_ptr + row_idx * input_row_stride
    input_addr = input_row + col_idx * input_col_stride
    x = tl.load(input_addr)
    
    sum = 0.0
    for k in range(K):
        term = 1.0 / (x + k) ** (n + 1)
        sum += term
    result = sign * factorial_n * sum
    output_row = output_ptr + row_idx * output_row_stride
    output_addr = output_row + col_idx * output_col_stride
    tl.store(output_addr, result)

def polygamma(n: int, input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    assert n >= 0, "n must be a nonnegative integer"
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda, "Input tensor must be on CUDA"
    
    *dims, size = input.shape
    input_flat = input.view(-1, size)
    M, N = input_flat.shape
    
    sign = (-1) ** (n + 1)
    factorial_n = math.factorial(n)
    K = 1000  # Number of terms in the series approximation
    
    grid = (M, N)
    _polygamma_kernel[grid](
        input_flat,
        out,
        input_flat.stride(0),
        input_flat.stride(1),
        out.stride(0),
        out.stride(1),
        n,
        sign,
        factorial_n,
        K,
        BLOCK_SIZE=128,
    )
    
    return out.view(*dims)
