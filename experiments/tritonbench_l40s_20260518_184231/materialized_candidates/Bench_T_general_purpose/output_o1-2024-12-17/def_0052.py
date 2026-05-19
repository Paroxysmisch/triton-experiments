import torch
import triton
import triton.language as tl

@triton.jit
def _sum_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to partial sum output
    N,  # number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # In-block reduction
    for stride in [BLOCK_SIZE // 2**i for i in range(1, 10)]:
        half = x[0:stride]
        x = x[0:stride] + x[stride:stride*2] if stride <= x.shape[0] // 2 else x
    # Store the block's partial sum
    tl.store(out_ptr + pid, x[0])

@triton.jit
def _sum_squares_kernel(
    x_ptr,  # pointer to input
    out_ptr,  # pointer to partial sum-of-squares output
    N,  # number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    # In-block reduction
    for stride in [BLOCK_SIZE // 2**i for i in range(1, 10)]:
        half = x_sq[0:stride]
        x_sq = x_sq[0:stride] + x_sq[stride:stride*2] if stride <= x_sq.shape[0] // 2 else x_sq
    # Store the block's partial sum of squares
    tl.store(out_ptr + pid, x_sq[0])

def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None):
    """
    Computes the sum of elements in the input tensor along the specified dimension(s),
    followed by calculating the standard
