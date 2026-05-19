import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, RSTD,
    stride_x, stride_y, stride_rstd,
    D, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    # Compute memory offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    
    # Load X
    x_ptrs = X + row * stride_x + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_x2 = tl.sum(x * x, axis=0)
    # Compute reciprocal of sqrt
    rstd = 1.0 / tl.sqrt(sum_x2)
    
    # Normalize and store
    y = x * rstd
    y_ptrs = Y + row * stride_y + col_offsets
    tl.store(y_ptrs, y, mask=mask)
    
    # Store rstd to RSTD (one value for this row)
    tl.store(RSTD + row * stride_rstd, rstd)

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, RSTD,
    stride_x, stride_dy, stride_dx, stride_rstd,
    D, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D
    
    x_ptrs = X + row *
