import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    # Pointers to matrices
    X_ptr, Y_ptr, OUT_ptr,
    # Matrix dimensions
    M, N,    
    # The stride variables represent how much to increase the ptr by when moving by 1 element
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Computes SwiGLU activation: out = x * sigmoid(x) * y
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block ID
    # This program is parallelized across M and N dimensions
    # We use a 2D grid where each block processes BLOCK_SIZE_M x BLOCK_SIZE_N elements
    bid_m = pid // (N // BLOCK_SIZE_N)
    bid_n = pid % (N // BLOCK_SIZE_N)

    # Start offset for this block
    start_m = bid_m * BLOCK_SIZE_M
    start_n = bid_n * BLOCK_SIZE_N

    # Initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask to handle edge cases
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Compute X and Y matrix offsets
    X_offs = offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    Y_offs = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    # Load data
    x = tl.load(X_ptr + X_offs, mask=mask)
    y = tl.load(Y_ptr + Y_offs, mask=mask)

    # Compute SwiGLU: x * sigmoid(x) * y
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x * y

    # Write output
    OUT_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(OUT_ptr + OUT_offs, out, mask=mask)

def _swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the SwiGLU forward kernel
    Args:
        x: Input tensor of shape (M, N)
        y: Input tensor of shape (M, N)
    Returns:
        out: Output tensor of shape (M, N)
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_contiguous(), "Input tensor x must be contiguous"
    assert y.is_contiguous(), "Input tensor y must be contiguous"
    
    M, N = x.shape
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Define block sizes (can be tuned for better performance)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out
