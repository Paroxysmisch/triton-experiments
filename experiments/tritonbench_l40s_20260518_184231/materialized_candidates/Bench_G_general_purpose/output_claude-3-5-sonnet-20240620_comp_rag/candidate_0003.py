import torch
import triton
import triton.language as tl

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
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = (A × B) * (A × B) using block-level matrix multiplication
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate per-block offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Calculate pointers for A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    temp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Create masks for bounds checking
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        # Compute matrix multiplication for this block
        temp_acc += tl.dot(a, b)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Square the result element-wise
    result = temp_acc * temp_acc

    # Write output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Calculate output pointers
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask for output bounds
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store the result
    tl.store(c_ptrs, result, mask=mask_c)

def matmul_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes C = (A × B) * (A × B) using blocks
    
    Args:
        a: tensor of shape (M, K)
        b: tensor of shape (K, N)
    Returns:
        c: tensor of shape (M, N)
    """
    assert a.ndim == b.ndim == 2, "Input tensors must be 2-dimensional"
    assert a.shape[1] == b.shape[0], "Invalid matrix dimensions for multiplication"
    
    # Extract dimensions
    M, K = a.shape
    _, N = b.shape
    
    # Ensure inputs are on GPU and have correct dtype
    a = a.cuda().to(torch.float32)
    b = b.cuda().to(torch.float32)
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Configure block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Create grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    matmul_squared_kernel[grid](
        c, a, b,
        M, N, K,
        c.stride(0), c.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return c
