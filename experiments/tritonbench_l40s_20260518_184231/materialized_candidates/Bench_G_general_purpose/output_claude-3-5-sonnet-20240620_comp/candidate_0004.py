import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    OUT_ptr, X_ptr, Y_ptr,
    stride_om, stride_on,  # Output strides
    stride_xm, stride_xn,  # X strides 
    stride_ym, stride_yn,  # Y strides
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Row and col block indices
    row_idx = pid // tl.cdiv(N, BLOCK_SIZE_N)
    col_idx = pid % tl.cdiv(N, BLOCK_SIZE_N)

    # Offsets
    offs_m = row_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = col_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask for bounds checking
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Pointers to current block
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    
    # Load data
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    
    # Compute SwiGLU: x * sigmoid(x) * y
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x * y
    
    # Store result
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)

def swiglu_forward(xy: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for SwiGLU activation.
    Args:
        xy: Input tensor of shape (batch_size, 2*features)
    Returns:
        Output tensor of shape (batch_size, features)
    """
    assert xy.dim() == 2, "Input must be 2-dimensional"
    assert xy.shape[1] % 2 == 0, "Feature dimension must be even"
    
    # Split input into x and y parts
    x, y = xy.chunk(2, dim=-1)
    
    # Make inputs contiguous if needed
    x = x.contiguous()
    y = y.contiguous()
    
    # Get dimensions
    M, N = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Configure meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    # Calculate grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        output, x, y,
        output.stride(0), output.stride(1),
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output
