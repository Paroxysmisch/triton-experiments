import triton
import triton.language as tl
import torch
import math

@triton.jit
def get_freq_multi_tokens(pos, dim, theta=10000.0):
    freq = pos / (theta ** (2.0 * (dim // 2) / dim))
    return freq

@triton.jit
def rbe_triton(
    x_ptr,  # pointer to input tensor [batch, M, K]
    out_ptr,  # pointer to output tensor [batch, M, K]
    batch,  # batch size
    M,  # sequence length dimension
    K,  # feature dimension
    stride_b,  # stride for batch dimension
    stride_m,  # stride for M dimension
    stride_k,  # stride for K dimension
    BLOCK_SIZE_M: tl.constexpr,  # block size in M dimension (2)
    BLOCK_SIZE_K: tl.constexpr,  # block size in K dimension (1024)
):
    # Program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    # Calculate offsets
    offs_m = (pid % (M // BLOCK_SIZE_M)) * BLOCK_SIZE_M
    offs_b = bid
    
    # Create mask for valid elements
    mask = tl.arange(0, BLOCK_SIZE_K) < K
    
    # Load input data
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + offs_b * stride_b + offs_m * stride_m + offs_k * stride_k
    
    # Load real and imaginary parts (even and odd indices)
    real = tl.load(x_ptrs + 0 * stride_k, mask=mask)
    imag = tl.load(x_ptrs + 1 * stride_k, mask=mask)
    
    # Calculate position-dependent frequencies
    pos = offs_m
    freq = get_freq_multi_tokens(pos, K // 2)
    
    # Calculate sin and cos values
    sin = tl.sin(freq)
    cos = tl.cos(freq)
    
    # Apply rotation
    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    
    # Store results
    out_ptrs = out_ptr + offs_b * stride_b + offs_m * stride_m + offs_k * stride_k
    tl.store(out_ptrs + 0 * stride_k, out_real, mask=mask)
    tl.store(out_ptrs + 1 * stride_k, out_imag, mask=mask)

def rbe_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the RBE Triton kernel
    Args:
        x: Input tensor of shape [batch, M, K]
    Returns:
        Transformed tensor of same shape
    """
    assert len(x.shape) == 3, "Input tensor must be 3D"
    batch, M, K = x.shape
    
    # Constants
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), batch)
    
    # Calculate strides
    stride_b = M * K
    stride_m = K
    stride_k = 1
    
    # Launch kernel
    rbe_triton[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        batch=batch,
        M=M,
        K=K,
        stride_b=stride_b,
        stride_m=stride_m,
        stride_k=stride_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out
