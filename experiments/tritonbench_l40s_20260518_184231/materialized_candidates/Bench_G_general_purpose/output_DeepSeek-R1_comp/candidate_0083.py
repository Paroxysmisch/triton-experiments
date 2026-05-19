import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X, W, Y,
    stride_d1, stride_d2, stride_n,
    w_stride_d2, w_stride_n,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get current program IDs for D1 and D2
    pid_d1 = tl.program_id(0)
    pid_d2 = tl.program_id(1)
    
    # Base pointers for X and W
    x_base = X + pid_d1 * stride_d1 + pid_d2 * stride_d2
    w_base = W + pid_d2 * w_stride_d2  # W has shape (D2, N)
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # First pass: compute sum and sum of squares
    for offset in range(0, N, BLOCK_SIZE):
        # Create mask to avoid out-of-bounds
        mask = offset + tl.arange(0, BLOCK_SIZE) < N
        # Load data from X
        x_ptr = x_base + offset * stride_n
        x = tl.load(x_ptr, mask=mask, other=0.0)
        # Update accumulators
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    
    # Compute mean and variance
    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize and apply weights
    for offset in range(0, N, BLOCK_SIZE):
        # Create mask
        mask = offset + tl.arange(0, BLOCK_SIZE) < N
        # Load X and W
        x_ptr = x_base + offset * stride_n
        w_ptr = w_base + offset * w_stride_n
        x = tl.load(x_ptr, mask=mask, other=0.0)
        w = tl.load(w_ptr, mask=mask, other=0.0)
        # Compute normalized value
        x_norm = (x - mean) * rstd
        y = x_norm * w
        # Store result in Y
        y_ptr = x_ptr - X + Y  # same strides as X
        tl.store(y_ptr, y, mask=mask)

def layernorm_forward(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-5):
    # Check input dimensions
    assert X.ndim == 3, "X must be 3D"
    assert W.ndim == 2, "W must be 2D"
    D1, D2, N = X.shape
    assert W.shape == (D2, N), f"W must have shape (D2, N) but got {W.shape}"
    
    # Allocate output tensor
    Y = torch.empty_like(X)
    
    # Compute strides for X
    stride_d1 = X.stride(0)
    stride_d2 = X.stride(1)
    stride_n = X.stride(2)
    
    # Compute strides for W
    w_stride_d2 = W.stride(0)
    w_stride_n = W.stride(1)
    
    # Grid dimensions (D1, D2)
    grid = (D1, D2)
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 1024  # can be tuned for optimal performance
    _layer_norm_fwd_kernel[grid](
        X, W, Y,
        stride_d1, stride_d2, stride_n,
        w_stride_d2, w_stride_n,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y
