import torch
import triton
import triton.language as tl
import math

@triton.jit
def rbe_triton(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    stride_b,  # batch stride
    stride_m,  # M dimension stride
    stride_k,  # K dimension stride
    batch,  # batch size
    M,  # M dimension size
    K,  # K dimension size
    BLOCK_SIZE_M: tl.constexpr,  # block size for M dimension (2)
    BLOCK_SIZE_K: tl.constexpr,  # block size for K dimension (1024)
):
    # Program ID for parallelization
    pid = tl.program_id(0)
    
    # Calculate block offsets
    num_blocks_m = tl.cdiv(M, BLOCK_SIZE_M)
    block_m = pid % num_blocks_m
    block_b = pid // num_blocks_m
    
    # Compute offsets for this block
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create mask for valid elements
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # Compute base pointer for this block
    base_ptr = x_ptr + block_b * stride_b
    
    # Load real and imaginary parts (even and odd indices)
    x_real_ptr = base_ptr + offs_m[:, None] * stride_m + (2 * offs_k[None, :]) * stride_k
    x_imag_ptr = base_ptr + offs_m[:, None] * stride_m + (2 * offs_k[None, :] + 1) * stride_k
    
    # Load data with masking
    real = tl.load(x_real_ptr, mask=mask_m[:, None] & mask_k[None, :])
    imag = tl.load(x_imag_ptr, mask=mask_m[:, None] & mask_k[None, :])
    
    # Compute position-dependent frequencies
    pos = offs_k[None, :]
    theta = 10000.0
    freq = tl.exp(-2.0 * pos * math.log(theta) / K)
    
    # Compute sin and cos
    sin = tl.sin(freq)
    cos = tl.cos(freq)
    
    # Apply rotary transformation
    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    
    # Store results
    out_real_ptr = out_ptr + block_b * stride_b + offs_m[:, None] * stride_m + (2 * offs_k[None, :]) * stride_k
    out_imag_ptr = out_ptr + block_b * stride_b + offs_m[:, None] * stride_m + (2 * offs_k[None, :] + 1) * stride_k
    
    # Store with masking
    tl.store(out_real_ptr, out_real, mask=mask_m[:, None] & mask_k[None, :])
    tl.store(out_imag_ptr, out_imag, mask=mask_m[:, None] & mask_k[None, :])

def rbe_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the RBE Triton kernel
    Args:
        x: Input tensor of shape [batch, M, K*2] where K*2 accounts for real and imaginary parts
    Returns:
        Output tensor of the same shape after applying rotary transformation
    """
    assert x.dim() == 3, "Input tensor must be 3-dimensional"
    batch, M, K = x.shape
    K = K // 2  # Actual feature dimension (half of tensor due to complex numbers)
    
    # Ensure tensor is contiguous and in the correct format
    x = x.contiguous()
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate grid size
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * batch,)
    
    # Launch kernel
    rbe_triton[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        stride_b=M * K * 2,
        stride_m=K * 2,
        stride_k=1,
        batch=batch,
        M=M,
        K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out
