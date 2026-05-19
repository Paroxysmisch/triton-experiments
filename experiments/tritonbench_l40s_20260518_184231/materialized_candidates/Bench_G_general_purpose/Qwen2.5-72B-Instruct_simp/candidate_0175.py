import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X,  # Input matrix
    DY,  # Gradients of the output
    DX,  # Output gradients
    N,  # Number of columns (features)
    eps,  # Small value for numerical stability
    stride_x_row,  # Stride for rows in X
    stride_x_col,  # Stride for columns in X
    stride_dy_row,  # Stride for rows in DY
    stride_dy_col,  # Stride for columns in DY
    stride_dx_row,  # Stride for rows in DX
    stride_dx_col,  # Stride for columns in DX
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load slices of X and DY
    x_slice = tl.load(X + row_idx * stride_x_row + col_idx * stride_x_col, mask=col_idx < N, other=0.0)
    dy_slice = tl.load(DY + row_idx * stride_dy_row + col_idx * stride_dy_col, mask=col_idx < N, other=0.0)

    # Compute the L2 norm of the row
    norm = tl.sqrt(tl.sum(x_slice * x_slice, axis=0) + eps)

    # Compute the variance
    variance = tl.sum(x_slice * dy_slice, axis=0) / norm

    # Compute the gradient
    dx_slice = (dy_slice * norm - x_slice * variance) / (norm * norm)

    # Store the computed gradient in DX
    tl.store(DX + row_idx * stride_dx_row + col_idx * stride_dx_col, dx_slice, mask=col_idx < N)

import torch

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.dim() == 2, "Input tensor must be 2D"
    assert dy.dim() == 2, "Gradient tensor must be 2D"
    assert x.shape == dy.shape, "Input and gradient tensors must have the same shape"

    M, N = x.shape
    dx = torch.empty_like(x)

    # Define block sizes
    BLOCK_SIZE = 128

    # Configure the grid and block sizes
    grid = (M,)

    # Launch the kernel
    _l2_norm_bwd_kernel[grid](
        x, dy, dx, N, eps,
        x.stride(0), x.stride(1),
        dy.stride(0), dy.stride(1),
        dx.stride(0), dx.stride(1),
        BLOCK_SIZE
    )

    return dx
