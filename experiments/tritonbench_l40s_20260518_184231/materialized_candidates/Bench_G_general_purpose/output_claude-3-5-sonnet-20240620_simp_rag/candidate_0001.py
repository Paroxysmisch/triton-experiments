import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(
    # Pointers to matrices
    X_ptr, Y_ptr, OUT_ptr,
    # Matrix dimensions
    M, N,
    # Strides for accessing memory
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    # Block sizes for parallel processing
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Computes SwiGLU forward pass: OUT = X * sigmoid(X) * Y
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Number of blocks in N dimension
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    # Block indices
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n
    
    # Starting indices for this block
    start_m = block_m * BLOCK_M
    start_n = block_n * BLOCK_N
    
    # Create offsets for this block
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    
    # Create mask to handle boundary conditions
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Compute memory offsets for matrices
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    
    # Load data
    x = tl.load(x_ptrs, mask=mask)
    y = tl.load(y_ptrs, mask=mask)
    
    # Compute SwiGLU: x * sigmoid(x) * y
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x * y
    
    # Store the result
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)

def _swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the SwiGLU forward kernel
    """
    # Make sure input tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()
    
    # Get dimensions
    M, N = x.shape
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    # Launch kernel
    _swiglu_fwd_kernel[grid](
        x.data_ptr(), y.data_ptr(), out.data_ptr(),
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N
    )
    
    return out
