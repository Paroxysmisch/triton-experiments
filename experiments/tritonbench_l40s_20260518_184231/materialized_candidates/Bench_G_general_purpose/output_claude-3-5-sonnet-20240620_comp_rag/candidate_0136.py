import triton
import triton.language as tl
import torch

@triton.jit
def squared_matmul_kernel(
    # Pointers to matrices
    C, A, B,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Kernel for computing C = (A @ B) * (A @ B)"""
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block pointers for the current block
    # Matrix A: [M, K]
    # Matrix B: [K, N]
    # Matrix C: [M, N]
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers to A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate to compute matrix multiplication A @ B
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Load A and B blocks with boundary checks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        # Compute matrix multiplication for this block
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Square the result (A @ B) * (A @ B)
    result = accumulator * accumulator
    
    # Write back the result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Write output with boundary checks
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=mask)

def squared_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = (A @ B) * (A @ B) using Triton
    
    Args:
        a: torch.Tensor of shape (M, K)
        b: torch.Tensor of shape (K, N)
    Returns:
        c: torch.Tensor of shape (M, N)
    """
    # Extract dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} and {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure meta-parameters
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Configure grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    squared_matmul_kernel[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return c
