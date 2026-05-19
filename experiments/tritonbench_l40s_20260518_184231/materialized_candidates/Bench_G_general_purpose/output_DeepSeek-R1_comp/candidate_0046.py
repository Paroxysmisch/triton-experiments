import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X_ptr,  # Pointer to input tensor
    Y_ptr,  # Pointer to output tensor
    stride_x_row,  # Stride between rows of X
    var_ptr,  # Pointer to variance tensor
    rstd_ptr,  # Pointer to reciprocal std tensor
    N,  # Number of columns (size of feature dimension)
    eps,  # Epsilon for numerical stability
    BLOCK: tl.constexpr,  # Block size (power of two)
):
    row_idx = tl.program_id(0)
    # Compute pointers to the current row in X and Y
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * N
    # Initialize sum of squares
    sum_sq = 0.0
    # Loop over elements in the row to compute sum of squares
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    # Compute variance and reciprocal standard deviation
    var_val = sum_sq
    rstd_val = 1.0 / tl.sqrt(var_val + eps)
    # Store variance and rstd
    tl.store(var_ptr + row_idx, var_val)
    tl.store(rstd_ptr + row_idx, rstd_val)
    # Normalize and store the output
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0)
        y = x * rstd_val
        tl.store(Y_row_ptr + cols, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X_ptr,  # Pointer to input tensor
    DY_ptr,  # Pointer to gradient of output
    DX_ptr,  # Pointer to gradient of input
    var_ptr,  # Pointer to variance tensor from forward pass
    rstd_ptr,  # Pointer to reciprocal std tensor from forward pass
    N,  # Number of columns (feature dimension size)
    BLOCK: tl.constexpr,  # Block size (power of two)
):
    row_idx = tl.program_id(0)
    # Compute pointers to the current row
    X_row_ptr = X_ptr + row_idx * N
    DY_row_ptr = DY_ptr + row_idx * N
    DX_row_ptr = DX_ptr + row_idx * N
    # Load rstd for the current row
    rstd_val = tl.load(rstd_ptr + row_idx)
    # Compute sum of DY * X for gradient calculation
    sum_dy_x = 0.0
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0)
        sum_dy_x += tl.sum(x * dy)
    # Compute coefficients for gradient calculation
    rstd_cubed = rstd_val * rstd_val * rstd_val
    coeff = rstd_cubed * sum_dy_x
    # Compute and store the gradient DX
    for offset in range(0, N, BLOCK):
        cols = offset + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_row_ptr + cols, mask=mask, other=0.0)
        dx = dy * rstd_val - x * coeff
        tl.store(DX_row_ptr + cols, dx, mask=mask)

def _l2_norm_fwd(X: torch.Tensor, eps: float):
    assert X.dim() == 2, "Input must be 2D"
    assert X.stride(-1) == 1, "Last dimension must be contiguous"
    rows, N = X.shape
    # Ensure feature dimension is within Triton's memory constraints
    max_feature_size = 16384  # 64KB / 4 bytes per float32
    assert N <= max_feature_size, f"Feature dimension must be <= {max_feature_size}"
    # Allocate output tensors
    Y = torch.empty_like(X)
    var = torch.empty((rows,), dtype=X.dtype, device=X.device)
    rstd = torch.empty((rows,), dtype=X.dtype, device=X.device)
    # Determine block size
    BLOCK = triton.next_power_of_2(N)
    if BLOCK < 128:
        BLOCK = 128  # Use minimum block size for better GPU utilization
    # Launch kernel
    grid = (rows,)
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, X.stride(0), var, rstd, N, eps, BLOCK=BLOCK
    )
    return Y, var, rstd

def _l2_norm_bwd(
    dy: torch.Tensor,  # Gradient of the output
    x: torch.Tensor,   # Input tensor from forward pass
    var: torch.Tensor,  # Variance from forward pass
    rstd: torch.Tensor, # Reciprocal std from forward pass
):
    assert dy.dim() == 2 and x.dim() == 2, "Inputs must be 2D"
    assert dy.shape == x.shape, "Shape mismatch between dy and x"
    rows, N = dy.shape
    assert var.shape == (rows,) and rstd.shape == (rows,), "Invalid var/rstd shape"
    assert dy.stride(-1) == 1 and x.stride(-1) == 1, "Contiguous last dimension required"
    # Allocate gradient tensor
    dx = torch.empty_like(dy)
    # Determine block size
    BLOCK = triton.next_power_of_2(N)
    if BLOCK < 128:
        BLOCK = 128
    # Launch kernel
    grid = (rows,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx, var, rstd, N, BLOCK=BLOCK
    )
    return dx
