import triton
import triton.language as tl
import torch


# Triton Kernel: L2 Normalization Backward Pass
@triton.jit
def _l2_norm_bwd_kernel(
    X_PTR, DY_PTR, DX_PTR,  # Pointers to input X, gradient DY, and output DX
    stride_x_row,  # Stride for accessing rows in X
    N,  # Number of elements in each row
    eps,  # Small constant for numerical stability
    BLOCK_N: tl.constexpr,  # Number of elements processed per block
):
    # Define row index for this kernel instance
    row_idx = tl.program_id(0)

    # Compute the start pointers for the current row
    x_ptr = X_PTR + row_idx * stride_x_row
    dy_ptr = DY_PTR + row_idx * stride_x_row
    dx_ptr = DX_PTR + row_idx * stride_x_row

    # Create a block of indices to process this row
    offsets = tl.arange(0, BLOCK_N)

    # Mask to ensure we stay within bounds of row length N
    mask = offsets < N

    # Load the input X and gradient DY for this row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

    # Compute the variance of X (mean is assumed to be zero)
    var = tl.sum(x * x, axis=0) / N

    # Compute the reciprocal of the standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)

    # Compute the intermediate term: sum(dy * x)
    dy_x_sum = tl.sum(dy * x, axis=0)

    # Compute the gradient dx using the formula
    dx = dy * rstd - dy_x_sum * (1.0 / (var + eps)) * rstd * x

    # Store the result back to DX
    tl.store(dx_ptr + offsets, dx, mask=mask)


# Wrapper Function
def _l2_norm_bwd(x, dy, eps=1e-5):
    """
    Backward pass for L2 normalization on a per-row basis.

    Args:
        x (torch.Tensor): Input tensor of shape (M, N).
        dy (torch.Tensor): Gradient tensor of shape (M, N).
        eps (float): Small constant for numerical stability.

    Returns:
        dx (torch.Tensor): Gradient of the input tensor, same shape as `x`.
    """
    # Validate input dimensions
    assert x.ndim == 2, "Input tensor `x` must be 2D (M, N)."
    assert dy.ndim == 2, "Gradient tensor `dy` must be 2D (M, N)."
    assert x.shape == dy.shape, "`x` and `dy` must have the same shape."

    # Get the shape of the input tensor
    M, N = x.shape

    # Determine BLOCK_N based on N
    BLOCK_N = 2 ** ((N - 1).bit_length())  # Next power of 2 of N
    if BLOCK_N > 2048:  # Limit the block size to avoid excessive memory usage
        raise ValueError("Feature dimension N exceeds the maximum allowable BLOCK_N (2048).")

    # Ensure the tensors are contiguous
    x = x.contiguous()
    dy = dy.contiguous()

    # Allocate output tensor
    dx = torch.empty_like(x)

    # Launch the Triton kernel
    grid = (M,)  # One kernel instance per row
    stride_x_row = x.stride(0)

    _l2_norm_bwd_kernel[grid](
        x, dy, dx,  # Input, gradient, and output pointers
        stride_x_row,  # Stride for row access
        N,  # Number of elements in each row
        eps,  # Epsilon for numerical stability
        BLOCK_N=BLOCK_N,  # Number of elements processed per block
    )

    # Return the output tensor
    return dx

# Input tensors
M, N = 128, 512
x = torch.randn(M, N, device='cuda', dtype=torch.float32)
dy = torch.randn(M, N, device='cuda', dtype=torch.float32)

# Backward pass for L2 normalization
dx = _l2_norm_bwd(x, dy)

print(dx.shape)  # Should match the shape of `x` and `dy`
