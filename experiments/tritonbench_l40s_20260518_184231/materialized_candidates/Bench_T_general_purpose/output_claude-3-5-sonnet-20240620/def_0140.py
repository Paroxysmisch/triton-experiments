import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scaling factors
    alpha, beta,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes C = beta * (alpha * tril(A) @ B)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    pid_n = pid % num_pid_in_group
    pid_m = group_id

    # Block start indices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_idx = k * BLOCK_SIZE_K + offs_k
        # Load A and B tiles
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + k_idx[None, :] * stride_ak,
                   mask=(offs_am[:, None] < M) & (k_idx[None, :] < K) & (offs_am[:, None] >= k_idx[None, :]),  # tril mask
                   other=0.0)
        b = tl.load(b_ptr + k_idx[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=(k_idx[:, None] < K) & (offs_bn[None, :] < N),
                   other=0.0)
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Scale by alpha and beta
    acc = acc * alpha * beta
    
    # Write back output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 acc,
                 mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Performs matrix multiplication between lower triangular part of A and B, with scaling.
    
    Args:
        A (torch.Tensor): Input matrix of shape (n, n)
        B (torch.Tensor): Input matrix of shape (n, p)
        alpha (float): Scaling factor for the matrix multiplication
        beta (float): Scaling factor for the final result
        
    Returns:
        torch.Tensor: Result matrix of shape (n, p)
    """
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D matrices"
    M, K = A.shape
    K, N = B.shape
    assert M == K, "Input matrix dimensions must match"
    
    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Configure meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    tril_mm_and_scale_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C
