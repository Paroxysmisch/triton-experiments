import triton
import triton.language as tl
import torch

@triton.jit
def matmul_squared_kernel(
    # Pointers to matrices
    C, A, B,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B 
    stride_cm, stride_cn,  # Strides for matrix C
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing C = (A @ B) * (A @ B)"""
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for A and B
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers to A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through k dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Bounds checking for k dimension
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Square the result element-wise
    acc = acc * acc
    
    # Write back result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking for output
    mask_m = offs_cm[:, None] < M
    mask_n = offs_cn[None, :] < N
    mask = mask_m & mask_n
    
    # Calculate output pointers
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    # Store result
    tl.store(c_ptrs, acc, mask=mask)

def matmul_squared(a: torch.Tensor, b: torch.Tensor):
    """
    Compute C = (A @ B) * (A @ B) using Triton
    
    Args:
        a: Input matrix A (M, K)
        b: Input matrix B (K, N)
    Returns:
        c: Output matrix C (M, N)
    """
    # Extract dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Block sizes (can be tuned)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_squared_kernel[grid](
        c, a, b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return c
