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
    stride_cm, stride_cn,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    # Block sizes (constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for A and B
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Calculate pointers for A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Bounds checking for k dimension
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        # Compute matrix multiplication for this block
        acc += tl.dot(a, b)
        
        # Move pointers to next k block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Square the result (element-wise multiplication with itself)
    acc = acc * acc

    # Write output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking for output
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Calculate output pointers
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    # Store result
    tl.store(c_ptrs, acc, mask=mask)

def matmul_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = (A × B) * (A × B) using Triton
    
    Args:
        a: Input matrix A (M × K)
        b: Input matrix B (K × N)
    
    Returns:
        Output matrix C (M × N)
    """
    # Check input dimensions
    assert a.dim() == 2 and b.dim() == 2, "Input matrices must be 2-dimensional"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions: {a.shape} and {b.shape}"

    # Ensure inputs are in correct format
    a = a.contiguous()
    b = b.contiguous()

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # Calculate grid dimensions
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul_squared_kernel[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return c
