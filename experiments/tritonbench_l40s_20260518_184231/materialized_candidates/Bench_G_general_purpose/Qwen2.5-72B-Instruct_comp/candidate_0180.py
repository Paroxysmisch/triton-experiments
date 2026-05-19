import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
):
    # Compute the row index for this program instance
    pid = tl.program_id(axis=0)
    row_start = pid * stride_x_row

    # Initialize pointers to the current row in X and Y
    X_row = X + row_start
    Y_row = Y + row_start

    # Load a block of data from X
    x_block = tl.load(X_row + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0)

    # Compute the sum of squares for variance
    variance = tl.sum(x_block * x_block, axis=0)

    # Compute the reciprocal of the square root of the variance plus eps
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Normalize the block and store the result in Y
    y_block = x_block * rstd
    tl.store(Y_row + tl.arange(0, BLOCK_N), y_block, mask=tl.arange(0, BLOCK_N) < N)

import torch
import triton
import triton.language as tl

def _l2_norm_fwd(x, eps=1e-5):
    # Ensure the input tensor is contiguous and in the correct shape
    x = x.contiguous()
    M, N = x.shape

    # Initialize an empty tensor to store the output
    y = torch.empty_like(x)

    # Calculate BLOCK_N based on x's element size and ensure it doesn't exceed 64KB
    element_size = x.element_size()
    max_block_size = 64 * 1024 // element_size
    BLOCK_N = min(N, max_block_size)

    if N > BLOCK_N:
        raise ValueError(f"Feature dimension N ({N}) is larger than BLOCK_N ({BLOCK_N})")

    # Launch the kernel
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x, y, N, eps, BLOCK_N
    )

    return y
