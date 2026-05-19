import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X_ptr,  # pointer to input tensor
    W_ptr,  # pointer to weight tensor
    Y_ptr,  # pointer to output tensor
    stride_xb, stride_xm, stride_xn,  # strides for accessing X
    stride_wb,  # stride for accessing W
    stride_yb, stride_ym, stride_yn,  # strides for accessing Y
    N,  # size of normalization dimension
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # number of elements to process per block
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch and row indices
    batch_id = pid // (stride_xm // stride_xn)
    row_id = pid % (stride_xm // stride_xn)
    
    # Compute pointers to the row for X and Y
    x_ptr = X_ptr + batch_id * stride_xb + row_id * stride_xn
    y_ptr = Y_ptr + batch_id * stride_yb + row_id * stride_yn
    
    # Load weights
    w_ptr = W_ptr + row_id * stride_wb
    weight = tl.load(w_ptr)
    
    # Compute mean
    mean = 0.0
    x_mask = tl.arange(0, BLOCK_SIZE) < N
    x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE), mask=x_mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    var = 0.0
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Normalize and apply weight
    rstd = 1 / tl.sqrt(var + eps)
    y = weight * (x_centered * rstd)
    
    # Store result
    tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y, mask=x_mask)

def layernorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """
    Apply layer normalization to the input tensor.
    
    Args:
        x: Input tensor of shape [B, M, N] where N is the normalization dimension
        weight: Weight tensor of shape [N]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor of shape [B, M, N]
    """
    batch_size, seq_len, hidden_dim = x.shape
    y = torch.empty_like(x)
    
    # Compute strides
    stride_xb = x.stride(0)
    stride_xm = x.stride(1)
    stride_xn = x.stride(2)
    stride_wb = weight.stride(0)
    stride_yb = y.stride(0)
    stride_ym = y.stride(1)
    stride_yn = y.stride(2)
    
    # Determine block size (round up to multiple of 32 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Launch kernel
    grid = (batch_size * seq_len,)
    _layer_norm_fwd_kernel[grid](
        x, weight, y,
        stride_xb, stride_xm, stride_xn,
        stride_wb,
        stride_yb, stride_ym, stride_yn,
        hidden_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y
