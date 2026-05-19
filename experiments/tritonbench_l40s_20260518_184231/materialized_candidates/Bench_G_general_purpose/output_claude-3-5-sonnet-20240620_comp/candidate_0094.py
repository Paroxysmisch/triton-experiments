import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
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
    """Matrix multiplication kernel: C = A @ B"""
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate starting position for this thread block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load sub-matrices from A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                   mask=(offs_am[:, None] < M) & (k + offs_k[None, :] < K),
                   other=0.0)
        
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=(k + offs_k[:, None] < K) & (offs_bn[None, :] < N),
                   other=0.0)
        
        # Compute matrix multiplication for this block
        acc += tl.dot(a, b)
    
    # Store result in C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Write back result
    c = tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 acc.to(tl.float16),
                 mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# Wrapper function
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix multiplication C = A @ B
    """
    # Check input dimensions
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    
    # Get matrix dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
