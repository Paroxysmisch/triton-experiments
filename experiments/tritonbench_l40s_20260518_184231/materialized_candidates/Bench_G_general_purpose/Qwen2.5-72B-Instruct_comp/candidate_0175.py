import triton
import triton.language as tl

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, stride_x_row, stride_dy_row, stride_dx_row, N, eps, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_N
    cols = tl.arange(0, BLOCK_N)

    # Load the row of X and DY
    x = tl.load(X + row_start + cols * stride_x_row, mask=cols < N, other=0.0)
    dy = tl.load(DY + row_start + cols * stride_dy_row, mask=cols < N, other=0.0)

    # Compute the variance of the input slice
    mean_x = tl.sum(x, axis=0) / N
    var = tl.sum(tl.square(x - mean_x), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Compute the gradient dx
    dx = dy * rstd - tl.sum(dy * x, axis=0) * (1 / (var + eps)) * rstd * x

    # Store the result in DX
    tl.store(DX + row_start + cols * stride_dx_row, dx, mask=cols < N)

import torch
import triton
import triton.language as tl

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
    # Ensure x and dy are contiguous and have the same shape
    assert x.shape == dy.shape, "Input and output gradient tensors must have the same shape"
    x = x.contiguous()
    dy = dy.contiguous()

    # Reshape the tensors to 2D (M, N)
    original_shape = x.shape
    M, N = x.numel() // N, N
    x = x.view(M, N)
    dy = dy.view(M, N)

    # Determine the block size
    BLOCK_N = min(triton.next_power_of_2(N), triton.config.MAX_FUSED_SIZE)
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds the maximum block size {BLOCK_N}")

    # Allocate memory for the output gradient
    dx = torch.empty_like(x)

    # Launch the kernel
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx, x.stride(0), dy.stride(0), dx.stride(0), N, eps, BLOCK_N
    )

    # Reshape the output back to the original shape
    dx = dx.view(original_shape)

    return dx
