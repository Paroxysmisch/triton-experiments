import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride elements (bytes)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Blocking parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for matrix multiplication C = A x B."""
    # Determine block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension by blocks
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k_off in range(num_k_blocks):
        # Compute K index for this block
        k = k_off * BLOCK_SIZE_K
        
        # Load A block with boundary checks
        a_offs_m = offs_m[:, None]
        a_offs_k = k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        a_mask = (a_offs_m < M) & (a_offs_k < K)
        a_block = tl.load(
            a_ptr + a_offs_m * stride_am + a_offs_k * stride_ak,
            mask=a_mask,
            other=0.0
        )
        
        # Load B block with boundary checks
        b_offs_k = k + tl.arange(0, BLOCK_SIZE_K)[:, None]
        b_offs_n = offs_n[None, :]
        b_mask = (b_offs_k < K) & (b_offs_n < N)
        b_block = tl.load(
            b_ptr + b_offs_k * stride_bk + b_offs_n * stride_bn,
            mask=b_mask,
            other=0.0
        )
        
        # Compute partial matrix multiplication
        accumulator += tl.dot(a_block, b_block, allow_tf32=True)
    
    # Store result with boundary checks
    c_offs_m = offs_m[:, None]
    c_offs_n = offs_n[None, :]
    c_mask = (c_offs_m < M) & (c_offs_n < N)
    c = accumulator.to(tl.float16)
    tl.store(
        c_ptr + c_offs_m * stride_cm + c_offs_n * stride_cn,
        c,
        mask=c_mask
    )

def matmul(a: torch.Tensor, b: torch.Tensor):
    """Compute matrix multiplication C = A x B using Triton."""
    # Check dimensions
    assert a.shape[1] == b.shape[0], "Dimension mismatch between A and B"
    M, K = a.shape
    _, N = b.shape
    
    # Preallocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Tuning parameters for blocking
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Grid dimensions using ceiling division
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
