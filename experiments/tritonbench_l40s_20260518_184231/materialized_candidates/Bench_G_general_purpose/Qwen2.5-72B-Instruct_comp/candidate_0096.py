import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(X, Y, W, stride_x_row, stride_x_col, stride_y_row, stride_y_col, stride_w, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Pointers to the current row of X and Y
    x_ptr = X + row_idx * stride_x_row + col_idx * stride_x_col
    y_ptr = Y + row_idx * stride_y_row + col_idx * stride_y_col

    # Load the weight vector
    w_ptr = W + col_idx * stride_w

    # Load the elements of the current row of X
    x = tl.load(x_ptr, mask=col_idx < N, other=0.0)

    # Compute the sum of squares
    sum_squares = tl.sum(x * x, axis=0)

    # Compute the mean of squares
    mean_squares = sum_squares / N

    # Compute the reciprocal of the standard deviation
    rstd = 1.0 / tl.sqrt(mean_squares + 1e-6)

    # Normalize the elements and multiply by the weight
    y = (x * rstd) * tl.load(w_ptr, mask=col_idx < N, other=1.0)

    # Store the result in Y
    tl.store(y_ptr, y, mask=col_idx < N)

import torch

def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor):
    assert x.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert x.size(-1) == weight.size(0), "Last dimension of x must match the size of weight"

    B, N = x.size(0), x.size(-1)
    BLOCK_SIZE = 128  # Adjust based on your GPU architecture
    num_warps = 4  # Adjust based on your GPU architecture

    # Ensure the tensors are on the same device
    device = x.device
    weight = weight.to(device)

    # Allocate output tensor
    y = torch.empty_like(x)

    # Launch the Triton kernel
    grid = (B,)
    _rms_norm_fwd_fused[grid](
        x, y, weight,
        x.stride(0), x.stride(-1),
        y.stride(0), y.stride(-1),
        weight.stride(0), N, BLOCK_SIZE
    )

    return y
