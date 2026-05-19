import torch
import triton
import triton.language as tl
import math

# Constants
THETA = 0.1  # Parameterized constant
BLOCK_SIZE_M = 2
BLOCK_SIZE_K = 1024

@triton.jit
def rbe_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    pos,  # Starting position
    batch,  # Batch size
    M,  # Size of M dimension
    K,  # Size of K dimension
    stride_xb,  # Stride for batch dimension of x
    stride_xm,  # Stride for M dimension of x
    stride_xk,  # Stride for K dimension of x
    stride_ob,  # Stride for batch dimension of output
    stride_om,  # Stride for M dimension of output
    stride_ok,  # Stride for K dimension of output
    BLOCK_SIZE_M: tl.constexpr,  # Block size for M dimension
    BLOCK_SIZE_K: tl.constexpr,  # Block size for K dimension
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Calculate offsets
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    # Create offsets for M and K dimensions
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

    # Create mask for bounds checking
    mask_m = offs_m < M
    mask_k = offs_k < K

    # Load input block
    x_ptrs = x_ptr + pid_b * stride_xb + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])

    # Compute position-dependent transformation
    pos_offset = pos + offs_k[None, :]
    theta_term = THETA * pos_offset
    cos_term = tl.cos(theta_term)
    sin_term = tl.sin(theta_term)
    
    # Apply transformation
    result = x_block * cos_term + x_block * sin_term

    # Store result
    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_k[None, :])

def rbe_triton_wrapper(x: torch.Tensor, pos: int) -> torch.Tensor:
    # Get tensor dimensions
    batch, M, K = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Compute strides
    stride_xb, stride_xm, stride_xk = x.stride()
    stride_ob, stride_om, stride_ok = out.stride()
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),  # Number of blocks in M dimension
        triton.cdiv(K, BLOCK_SIZE_K),  # Number of blocks in K dimension
        batch,                         # Batch dimension
    )
    
    # Launch kernel
    rbe_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        pos=pos,
        batch=batch,
        M=M,
        K=K,
        stride_xb=stride_xb,
        stride_xm=stride_xm,
        stride_xk=stride_xk,
        stride_ob=stride_ob,
        stride_om=stride_om,
        stride_ok=stride_ok,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return out
