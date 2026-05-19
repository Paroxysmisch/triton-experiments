import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,  # Pointer to input tensor
    Y,  # Pointer to output tensor
    Rstd,  # Pointer to reciprocal std dev tensor
    stride_x_row,  # Stride between rows of X
    N,  # Number of columns (features per row)
    eps,  # Small epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size for computation
):
    # Each program processes a single row
    row_idx = tl.program_id(0)
    X_row = X + row_idx * stride_x_row
    Y_row = Y + row_idx * stride_x_row

    # Compute sum of squares for the row using multiple blocks
    sum_sq = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)
    
    # Compute reciprocal of L2 norm (rstd) with epsilon
    rstd = 1.0 / tl.sqrt(sum_sq + eps)
    tl.store(Rstd + row_idx, rstd)

    # Normalize and store the output
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0)
        y = x * rstd
        tl.store(Y_row + cols, y, mask=mask)

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, Rstd, DX,
    stride_x_row, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_row = X + row_idx * stride_x_row
    DY_row = DY + row_idx * stride_x_row
    DX_row = DX + row_idx * stride_x_row

    # Load precomputed rstd for this row
    rstd = tl.load(Rstd + row_idx)

    # Compute sum(X * DY) for gradient calculations
    sum_xdy = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0)
        dy = tl.load(DY_row + cols, mask=mask, other=0.0)
        sum_xdy += tl.sum(x * dy)

    # Precompute rstd^3 for efficiency
    rstd_cubed = rstd * rstd * rstd

    # Compute and store gradients
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0)
        dy = tl.load(DY_row + cols, mask=mask, other=0.0)
        dx = rstd * dy - rstd_cubed * x * sum_xdy
        tl.store(DX_row + cols, dx, mask=mask)

def _l2_norm_fwd(X: torch.Tensor, eps: float = 1e-5):
    # Ensure input is 2D and contiguous
    assert X.is_contiguous() and X.dim() == 2, "Input must be 2D contiguous"
    M, N = X.shape
    
    # Check feature dimension limit (64KB)
    max_feature_dim = 16384  # 64KB / 4 bytes (float32)
    if N > max_feature_dim:
        raise ValueError(f"Feature dimension {N} exceeds 64KB limit")
    
    # Allocate output tensors
    Y = torch.empty_like(X)
    Rstd = torch.empty(M, dtype=torch.float32, device=X.device)
    
    # Configure kernel launch parameters
    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Launch forward kernel
    _l2_norm_fwd_1pass_kernel[grid](X, Y, Rstd, X.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y, Rstd

def _l2_norm_bwd(
    grad_output: torch.Tensor,
    X: torch.Tensor,
    Rstd: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    # Validate inputs and allocate gradient tensor
    assert grad_output.is_contiguous() and X.is_contiguous(), "Inputs must be contiguous"
    assert grad_output.dim() == 2, "Gradient must be 2D"
    M, N = grad_output.shape
    
    max_feature_dim = 16384
    if N > max_feature_dim:
        raise ValueError(f"Feature dimension {N} exceeds 64KB limit")
    
    grad_input = torch.empty_like(X)
    
    # Configure kernel launch parameters
    grid = (M,)
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Launch backward kernel
    _l2_norm_bwd_kernel[grid](
        X, grad_output, Rstd, grad_input,
        X.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return grad_input
