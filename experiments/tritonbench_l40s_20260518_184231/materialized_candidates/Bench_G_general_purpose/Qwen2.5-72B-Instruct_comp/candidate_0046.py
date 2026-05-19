import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, stride_x_col, stride_y_row, stride_y_col, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Load the row of data
    x = tl.load(X + row * stride_x_row + cols * stride_x_col, mask=cols < n_cols, other=0.0)

    # Compute the sum of squares (variance)
    variance = tl.sum(x * x, axis=0)

    # Compute the reciprocal square root (rstd)
    rstd = 1.0 / tl.sqrt(variance + 1e-6)

    # Normalize the data
    y = x * rstd

    # Store the result
    tl.store(Y + row * stride_y_row + cols * stride_y_col, y, mask=cols < n_cols)

@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x_row, stride_x_col, stride_dy_row, stride_dy_col, stride_dx_row, stride_dx_col, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Load the inputs and gradients
    x = tl.load(X + row * stride_x_row + cols * stride_x_col, mask=cols < n_cols, other=0.0)
    dy = tl.load(DY + row * stride_dy_row + cols * stride_dy_col, mask=cols < n_cols, other=0.0)

    # Compute the sum of squares (variance)
    variance = tl.sum(x * x, axis=0)

    # Compute the reciprocal square root (rstd)
    rstd = 1.0 / tl.sqrt(variance + 1e-6)

    # Compute the gradient of the loss with respect to the input
    dx = dy * rstd - (tl.sum(dy * x * rstd, axis=0) * x * rstd * rstd)

    # Store the result
    tl.store(DX + row * stride_dx_row + cols * stride_dx_col, dx, mask=cols < n_cols)

import torch

def _l2_norm_fwd(X: torch.Tensor, Y: torch.Tensor):
    assert X.is_cuda and Y.is_cuda, "Tensors must be on GPU"
    assert X.shape == Y.shape, "Input and output tensors must have the same shape"
    assert X.stride(-1) == 1 and Y.stride(-1) == 1, "Last dimension stride must be 1"

    n_rows, n_cols = X.shape
    assert n_cols < 64 * 1024, "Feature dimension size must be less than 64KB"

    BLOCK_SIZE = 128
    grid = (n_rows,)

    _l2_norm_fwd_1pass_kernel[grid](
        X, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        n_cols, BLOCK_SIZE
    )

def _l2_norm_bwd(X: torch.Tensor, DY: torch.Tensor, DX: torch.Tensor):
    assert X.is_cuda and DY.is_cuda and DX.is_cuda, "Tensors must be on GPU"
    assert X.shape == DY.shape == DX.shape, "Input, gradient, and output tensors must have the same shape"
    assert X.stride(-1) == 1 and DY.stride(-1) == 1 and DX.stride(-1) == 1, "Last dimension stride must be 1"

    n_rows, n_cols = X.shape
    assert n_cols < 64 * 1024, "Feature dimension size must be less than 64KB"

    BLOCK_SIZE = 128
    grid = (n_rows,)

    _l2_norm_bwd_kernel[grid](
        X, DY, DX,
        X.stride(0), X.stride(1),
        DY.stride(0), DY.stride(1),
        DX.stride(0), DX.stride(1),
        n_cols, BLOCK_SIZE
    )

import torch

# Create a random input tensor
X = torch.randn(1024, 128, device='cuda')

# Allocate output tensor
Y = torch.empty_like(X)

# Forward pass
_l2_norm_fwd(X, Y)

# Create random gradients for the backward pass
DY = torch.randn_like(X)

# Allocate gradient tensor for the input
DX = torch.empty_like(X)

# Backward pass
_l2_norm_bwd(X, DY, DX)
