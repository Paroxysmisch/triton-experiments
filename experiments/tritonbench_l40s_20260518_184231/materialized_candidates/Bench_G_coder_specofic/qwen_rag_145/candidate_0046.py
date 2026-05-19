import triton.language as tl
import numpy as np

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, row_stride, num_cols, eps
):
    """
    An in-place implementation of L2 normalization.
    """
    # Identify the current row and column
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Compute the base index for the current row
    base_idx = row * row_stride
    X_slice = X + base_idx
    Y_slice = Y + base_idx

    # Compute square sum across columns
    sum_sq = 0
    for off in range(0, num_cols, tl.num_warps):
        cols = off + tl.arange(0, tl.num_warps)
        mask = cols < num_cols
        x = tl.load(X_slice + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x
    sum_sq_sqrt = tl.sqrtd(sum_sq)
    norm_factor = sum_sq_sqrt / np.sqrt(num_cols) + eps

    for off in range(0, num_cols, tl.num_warps):
        cols = off + tl.arange(0, tl.num_warps)
        mask = cols < num_cols
        x = tl.load(X_slice + cols, mask=mask, other=0.0).to(tl.float32)
        y = x / norm_factor
        tl.store(Y_slice + cols, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, row_stride, num_cols, eps
):
    """
    The backward pass for L2 normalization.
    """
    # Identify the current row and column
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Compute the base index for the current row
    base_idx = row * row_stride
    X_slice = X + base_idx
    DY_slice = DY + base_idx
    DX_slice = DX + base_idx

    # Compute square sum across columns
    sum_sq = 0
    for off in range(0, num_cols, tl.num_warps):
        cols = off + tl.arange(0, tl.num_warps)
        mask = cols < num_cols
        x = tl.load(X_slice + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x
    sum_sq_sqrt = tl.sqrtd(sum_sq)
    norm_factor = sum_sq_sqrt / np.sqrt(num_cols) + eps

    for off in range(0, num_cols, tl.num_warps):
        cols = off + tl.arange(0, tl.num_warps)
        mask = cols < num_cols
        x = tl.load(X_slice + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_slice + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dy * x / (norm_factor ** 2) - dy * x * sum_sq / (norm_factor ** 3 * num_cols)
        tl.store(DX_slice + cols, dx, mask=mask)
