import triton
import triton.language as tl
import torch

@triton.jit
def matmul_square_kernel(
    # Pointers to matrices
    C, A, B,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    # Block sizes (constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Kernel for computing C = (A x B) * (A x B)"""
    
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for A and B matrices
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Calculate pointers for A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator for first matrix multiplication
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # First matrix multiplication: temp = A x B
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_idx, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_idx, other=0.0)
        # Compute matrix multiplication
        accumulator += tl.dot(a, b)
        # Move pointers to next block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Square the result element-wise
    result = accumulator * accumulator

    # Write output to C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Store the result with bounds checking
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=mask)

# Wrapper function
def matmul_square(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = (A x B) * (A x B) using Triton
    
    Args:
        a: Input matrix A (M x K)
        b: Input matrix B (K x N)
    Returns:
        Output matrix C (M x N)
    """
    # Extract dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {K} != {K_}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    matmul_square_kernel[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c
