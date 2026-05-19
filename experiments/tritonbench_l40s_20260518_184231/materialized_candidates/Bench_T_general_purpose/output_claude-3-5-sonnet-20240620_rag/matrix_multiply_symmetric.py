import triton
import triton.language as tl
import torch
import math

@triton.jit
def matrix_multiply_symmetric_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Constants
    alpha, beta,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,  
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Kernel for performing fused matrix multiplication and symmetric update:
    C = alpha * (A @ B) + beta * C
    C = alpha * (C @ C.T) + beta * C
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # First matrix multiplication: C = alpha * (A @ B) + beta * C
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load C for beta * C
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    c_curr = tl.load(c_ptrs, mask=mask)
    
    # Compute alpha * (A @ B) + beta * C
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N)
        acc += tl.dot(a, b)
    
    c_new = alpha * acc + beta * c_curr
    tl.store(c_ptrs, c_new, mask=mask)
    
    # Synchronize to ensure first multiplication is complete
    tl.debug_barrier()
    
    # Second matrix multiplication: C = alpha * (C @ C.T) + beta * C
    # Load updated C
    c_curr = tl.load(c_ptrs, mask=mask)
    
    # Initialize accumulator for second multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute C @ C.T
    for k in range(0, N, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k
        c_ptrs_k = c_ptr + offs_am[:, None] * stride_cm + offs_k[None, :] * stride_cn
        ct_ptrs_k = c_ptr + offs_k[:, None] * stride_cm + offs_bn[None, :] * stride_cn
        
        mask_k = (offs_k < N)
        c1 = tl.load(c_ptrs_k, mask=(offs_am[:, None] < M) & (mask_k[None, :]))
        c2 = tl.load(ct_ptrs_k, mask=(mask_k[:, None]) & (offs_bn[None, :] < N))
        acc += tl.dot(c1, c2)
    
    # Final update
    c_final = alpha * acc + beta * c_curr
    tl.store(c_ptrs, c_final, mask=mask)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, 
                            alpha: float, beta: float) -> torch.Tensor:
    """
    Computes two sequential matrix operations:
    1. C = alpha * (A @ B) + beta * C
    2. C = alpha * (C @ C.T) + beta * C
    
    Args:
        A: Input matrix of shape (M, K)
        B: Input matrix of shape (K, N)
        C: Input/output matrix of shape (M, N)
        alpha: Scalar multiplier for matrix products
        beta: Scalar multiplier for C
    
    Returns:
        Updated C tensor
    """
    M, K = A.shape
    K_, N = B.shape
    assert K == K_, f"Incompatible dimensions: A: {A.shape}, B: {B.shape}"
    assert C.shape == (M, N), f"Incompatible dimensions: C: {C.shape}, expected: ({M}, {N})"
    
    # Handle small matrices with PyTorch
    if M * N <= 1024:
        C = alpha * torch.mm(A, B) + beta * C
        C = alpha * torch.mm(C, C.T) + beta * C
        return C
    
    # Kernel parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * 
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Run kernel
    matrix_multiply_symmetric_kernel[grid](
        A, B, C,
        M, N, K,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return C
