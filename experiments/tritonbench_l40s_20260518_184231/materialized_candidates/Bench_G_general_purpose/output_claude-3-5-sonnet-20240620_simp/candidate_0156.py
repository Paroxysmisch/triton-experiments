import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    Y, X, M,  # Pointers to tensors
    stride_xb, stride_xh, stride_xm,  # Strides for input X
    stride_yb, stride_yh, stride_ym,  # Strides for output Y
    stride_mb, stride_mh, stride_mm,  # Strides for mask M
    n_cols,  # Size of the last dimension
    log_output,  # Whether to apply log
    is_causal,  # Whether to apply causal mask
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head indices
    batch_idx = pid // stride_xh
    head_idx = pid % stride_xh
    
    # Compute base pointers
    x_ptr = X + batch_idx * stride_xb + head_idx * stride_xm
    y_ptr = Y + batch_idx * stride_yb + head_idx * stride_ym
    m_ptr = M + batch_idx * stride_mb + head_idx * stride_mm if M is not None else None
    
    # Initialize row offset
    row_start = 0
    row_end = n_cols
    
    # Load input elements
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    x = tl.load(x_ptr + col, mask=mask, other=-float('inf'))
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = col <= row_start
        x = tl.where(causal_mask, x, -float('inf'))
    
    # Apply attention mask if provided
    if M is not None:
        m = tl.load(m_ptr + col, mask=mask, other=0)
        x = x + m
    
    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    
    # Compute exponentials
    x = x - x_max
    numerator = tl.exp(x)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    y = numerator / denominator
    
    # Apply log if needed
    if log_output:
        y = tl.log(y)
    
    # Store output
    tl.store(y_ptr + col, y, mask=mask)

@triton.jit
def _softmax_backward_kernel(
    DX, DY, Y,  # Pointers to tensors
    stride_dxb, stride_dxh, stride_dxm,  # Strides for grad input
    stride_dyb, stride_dyh, stride_dym,  # Strides for grad output
    stride_yb, stride_yh, stride_ym,  # Strides for softmax output
    n_cols,  # Size of the last dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head indices
    batch_idx = pid // stride_dyh
    head_idx = pid % stride_dyh
    
    # Compute base pointers
    dx_ptr = DX + batch_idx * stride_dxb + head_idx * stride_dxm
    dy_ptr = DY + batch_idx * stride_dyb + head_idx * stride_dym
    y_ptr = Y + batch_idx * stride_yb + head_idx * stride_ym
    
    # Load elements
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    
    dy = tl.load(dy_ptr + col, mask=mask, other=0.0)
    y = tl.load(y_ptr + col, mask=mask, other=0.0)
    
    # Compute gradient
    sum_dy_y = tl.sum(dy * y, axis=0)
    dx = y * (dy - sum_dy_y)
    
    # Store result
    tl.store(dx_ptr + col, dx, mask=mask)

# Python wrapper for forward pass
def softmax(x, mask=None, log_output=False, is_causal=False):
    batch_size, n_heads, n_cols = x.shape
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Configure grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size * n_heads,)
    
    # Launch kernel
    _softmax_kernel[grid](
        y, x, mask,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        mask.stride(0) if mask is not None else 0,
        mask.stride(1) if mask is not None else 0,
        mask.stride(2) if mask is not None else 0,
        n_cols,
        log_output,
        is_causal,
        BLOCK_SIZE,
        num_warps=4,
    )
    
    return y

# Python wrapper for backward pass
def softmax_backward(grad_output, output):
    batch_size, n_heads, n_cols = grad_output.shape
    
    # Allocate gradient input
    grad_input = torch.empty_like(grad_output)
    
    # Configure grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size * n_heads,)
    
    # Launch kernel
    _softmax_backward_kernel[grid](
        grad_input, grad_output, output,
        grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
        grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        n_cols,
        BLOCK_SIZE,
        num_warps=4,
    )
    
    return grad_input
