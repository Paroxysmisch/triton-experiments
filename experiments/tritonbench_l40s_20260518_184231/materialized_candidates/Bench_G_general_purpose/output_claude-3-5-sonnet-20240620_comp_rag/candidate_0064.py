import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    # Pointers to tensors
    X_ptr,          # Input tensor pointer
    W_ptr,          # Weight tensor pointer
    Y_ptr,          # Output tensor pointer
    # Tensor strides
    stride_b,       # Batch stride
    stride_m,       # Row stride
    stride_n,       # Column stride
    w_stride,       # Weight stride
    # Dimensions and constants
    N: tl.constexpr,        # Size of normalization dimension
    eps: tl.constexpr,      # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr # Block size for parallel processing
):
    # Get program ID for batch and row
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Calculate base offset for current row
    row_offset = pid_b * stride_b + pid_m * stride_m
    
    # Create offset range for block processing
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Initialize variance accumulator
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute variance
    for block_start in range(0, N, BLOCK_SIZE):
        offs_n = block_start + offs
        mask = offs_n < N
        
        # Load and square values
        x = tl.load(X_ptr + row_offset + offs_n * stride_n, mask=mask, other=0.0)
        x = x.to(tl.float32)
        var += x * x * mask
    
    # Compute RMS statistics
    var = tl.sum(var) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize and apply weights
    for block_start in range(0, N, BLOCK_SIZE):
        offs_n = block_start + offs
        mask = offs_n < N
        
        # Load input and weights
        x = tl.load(X_ptr + row_offset + offs_n * stride_n, mask=mask, other=0.0)
        w = tl.load(W_ptr + offs_n * w_stride, mask=mask, other=0.0)
        
        # Normalize and scale
        x = x.to(tl.float32)
        y = (x * rrms) * w
        
        # Store result
        tl.store(Y_ptr + row_offset + offs_n * stride_n, y, mask=mask)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply RMS normalization to input tensor
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_dim]
        weight: Weight tensor of shape [hidden_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor of same shape as input
    """
    batch_size, seq_len, hidden_dim = x.shape
    y = torch.empty_like(x)
    
    # Launch kernel
    grid = (batch_size, seq_len)
    rms_norm_kernel[grid](
        x, weight, y,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0),
        hidden_dim, eps,
        BLOCK_SIZE=min(hidden_dim, 4096),
        num_warps=4
    )
    
    return y
