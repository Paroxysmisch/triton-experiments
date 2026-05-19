import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX,  # pointers to tensors
    stride_x_row,  # stride for accessing rows
    N,  # number of elements per row
    M,  # number of rows
    eps,  # epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # number of elements to process per block
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets for this row
    row_start = pid * stride_x_row
    
    # Create mask for valid elements
    mask = tl.arange(0, BLOCK_N) < N
    
    # Load input and gradient
    x = tl.load(X + row_start + tl.arange(0, BLOCK_N), mask=mask)
    dy = tl.load(DY + row_start + tl.arange(0, BLOCK_N), mask=mask)
    
    # Compute variance
    mean_x2 = tl.sum(x * x, axis=0) / N
    var = mean_x2
    
    # Compute reciprocal of standard deviation
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Compute dy * x
    dy_x = dy * x
    
    # Compute sum(dy * x)
    sum_dy_x = tl.sum(dy_x, axis=0) / N
    
    # Compute final gradient
    dx = (dy * rstd - sum_dy_x * rstd * x)
    
    # Store result
    tl.store(DX + row_start + tl.arange(0, BLOCK_N), dx, mask=mask)

def _l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Backward pass for L2 normalization.
    
    Args:
        x: Input tensor of shape (*, N) where * is any number of dimensions
        dy: Gradient tensor of same shape as x
        eps: Small constant for numerical stability
    
    Returns:
        dx: Input gradient tensor of same shape as x
    """
    # Reshape input into 2D tensor
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    dy = dy.reshape(-1, x_shape[-1])
    
    # Get dimensions
    M, N = x.shape
    
    # Determine block size (next power of 2, up to maximum allowed size)
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N > 2048:  # Maximum allowed block size
        raise ValueError(f"Input feature dimension ({N}) too large. Maximum allowed is 2048.")
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    dy = dy.contiguous()
    
    # Allocate output
    dx = torch.empty_like(x)
    
    # Launch kernel
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        x, dy, dx,
        x.stride(0),  # stride between rows
        N, M, eps,
        BLOCK_N=BLOCK_N,
    )
    
    # Reshape output back to original shape
    return dx.reshape(x_shape)
