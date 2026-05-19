import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr,                     # Pointers to input, weight and output tensors
    stride_xd1, stride_xd2, stride_xn,       # Strides for input X
    stride_wd2, stride_wn,                   # Strides for weight W
    stride_yd1, stride_yd2, stride_yn,       # Strides for output Y
    D1, D2, N,                              # Dimensions
    BLOCK_SIZE: tl.constexpr                 # Block size for optimization
):
    # Get program ID
    pid_d1 = tl.program_id(0)
    pid_d2 = tl.program_id(1)
    
    # Compute the starting offset for this block
    x_offset = pid_d1 * stride_xd1 + pid_d2 * stride_xd2
    w_offset = pid_d2 * stride_wd2
    y_offset = pid_d1 * stride_yd1 + pid_d2 * stride_yd2
    
    # Initialize accumulators for mean and variance
    mean = 0.0
    m2 = 0.0  # For variance computation
    
    # First pass: compute mean
    for idx in range(0, N, BLOCK_SIZE):
        # Create block mask
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        # Load input values
        x = tl.load(X_ptr + x_offset + idx * stride_xn, mask=mask, other=0.0)
        # Accumulate sum
        mean += tl.sum(x * mask, axis=0)
    
    mean = mean / N
    
    # Second pass: compute variance
    for idx in range(0, N, BLOCK_SIZE):
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        x = tl.load(X_ptr + x_offset + idx * stride_xn, mask=mask, other=0.0)
        diff = x - mean
        m2 += tl.sum((diff * diff) * mask, axis=0)
    
    var = m2 / N
    rstd = 1 / tl.sqrt(var + 1e-5)
    
    # Third pass: normalize and scale
    for idx in range(0, N, BLOCK_SIZE):
        mask = idx + tl.arange(0, BLOCK_SIZE) < N
        # Load input and weights
        x = tl.load(X_ptr + x_offset + idx * stride_xn, mask=mask, other=0.0)
        w = tl.load(W_ptr + w_offset + idx * stride_wn, mask=mask, other=0.0)
        
        # Normalize and scale
        y = (x - mean) * rstd * w
        
        # Store result
        tl.store(Y_ptr + y_offset + idx * stride_yn, y, mask=mask)

def layernorm_forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for layer normalization.
    
    Args:
        x: Input tensor of shape (D1, D2, N)
        weight: Weight tensor of shape (D2, N)
    
    Returns:
        y: Output tensor of shape (D1, D2, N)
    """
    # Get dimensions
    D1, D2, N = x.shape
    assert weight.shape == (D2, N), f"Weight shape {weight.shape} doesn't match expected shape ({D2}, {N})"
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Compute strides
    stride_xd1, stride_xd2, stride_xn = x.stride()
    stride_wd2, stride_wn = weight.stride()
    stride_yd1, stride_yd2, stride_yn = y.stride()
    
    # Define block size (can be tuned)
    BLOCK_SIZE = 32
    
    # Launch kernel
    grid = (D1, D2)
    _layer_norm_fwd_kernel[grid](
        x, weight, y,
        stride_xd1, stride_xd2, stride_xn,
        stride_wd2, stride_wn,
        stride_yd1, stride_yd2, stride_yn,
        D1, D2, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y
