import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X,  # Input tensor pointer
    W,  # Weight tensor pointer
    Y,  # Output tensor pointer
    stride_x_N, stride_x_hn, stride_x_hd,  # Strides for X
    stride_y_N, stride_y_hn, stride_y_hd,  # Strides for Y
    stride_w_hn, stride_w_hd,  # Strides for W
    N,  # Size of normalization dimension
    eps,  # Numerical stability term
    BLOCK_SIZE: tl.constexpr,  # Processing block size
):
    # Get 2D position in grid (sequence, head)
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Calculate base pointers for current slice
    x_ptr = X + seq_id * stride_x_N + head_id * stride_x_hn
    y_ptr = Y + seq_id * stride_y_N + head_id * stride_y_hn
    w_ptr = W + head_id * stride_w_hn

    # Phase 1: Compute mean using Welford's algorithm
    mean = 0.0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean) / N  # Total mean for the slice

    # Phase 2: Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        x_centered = tl.where(mask, x - mean, 0.0)
        _var += x_centered * x_centered
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)  # Reciprocal of std dev

    # Phase 3: Normalize and scale
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        # Load data with proper stride handling
        x = tl.load(x_ptr + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(w_ptr + cols * stride_w_hd, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize and apply weights
        x_hat = (x - mean) * rstd
        y = x_hat * weight
        
        # Store with output stride consideration
        tl.store(y_ptr + cols * stride_y_hd, y.to(X.dtype.element_ty), mask=mask)

def layernorm_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    eps: float
) -> torch.Tensor:
    # Validate input dimensions
    assert X.ndim == 3, "Input tensor must be 3D"
    assert W.ndim == 2, "Weight tensor must be 2D"
    assert X.shape[-2:] == W.shape, "Feature dimension mismatch between X and W"

    # Initialize output tensor
    Y = torch.empty_like(X)
    
    # Calculate tensor strides
    stride_x_N, stride_x_hn, stride_x_hd = X.stride()
    stride_y_N, stride_y_hn, stride_y_hd = Y.stride()
    stride_w_hn, stride_w_hd = W.stride()

    # Kernel configuration
    N = X.size(-1)
    BLOCK_SIZE = 128  # Tuned for modern GPU architectures
    grid = (X.shape[0], X.shape[1])  # Map each (sequence, head) to a kernel instance

    # Launch kernel with corrected stride handling
    _layer_norm_fwd_kernel[grid](
        X, W, Y,
        stride_x_N, stride_x_hn, stride_x_hd,
        stride_y_N, stride_y_hn, stride_y_hd,
        stride_w_hn, stride_w_hd,
        N, eps, BLOCK_SIZE
    )
    
    return Y
