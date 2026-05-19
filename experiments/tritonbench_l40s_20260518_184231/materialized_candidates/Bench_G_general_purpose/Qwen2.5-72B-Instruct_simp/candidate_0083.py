import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(X, W, Y, stride_x_row, stride_x_col, stride_w, stride_y_row, stride_y_col, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Compute the start and end indices for the row
    start_x = row_idx * stride_x_row
    start_y = row_idx * stride_y_row

    # Load the input data for the current row
    x = tl.load(X + start_x + col_idx * stride_x_col, mask=col_idx < N, other=0.0).to(tl.float32)

    # Compute the mean
    mean = tl.sum(x, axis=0) / N

    # Compute the variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Compute the normalized values
    rstd = 1.0 / tl.sqrt(var + eps)
    x_normalized = x_centered * rstd

    # Load the weights
    w = tl.load(W + col_idx * stride_w, mask=col_idx < N, other=0.0).to(tl.float32)

    # Apply the weights
    y = x_normalized * w

    # Store the output
    tl.store(Y + start_y + col_idx * stride_y_col, y, mask=col_idx < N)

import torch

def layernorm_forward(X, W, eps=1e-5):
    # Get the shape of the input tensor
    B, N = X.shape

    # Allocate the output tensor
    Y = torch.empty_like(X)

    # Compute the strides
    stride_x_row = N
    stride_x_col = 1
    stride_w = 1
    stride_y_row = N
    stride_y_col = 1

    # Define the grid and block sizes
    grid = (B, (N + 1024 - 1) // 1024)
    block = 1024

    # Launch the kernel
    _layer_norm_fwd_kernel[grid, block](
        X, W, Y, stride_x_row, stride_x_col, stride_w, stride_y_row, stride_y_col, N, eps
    )

    return Y
