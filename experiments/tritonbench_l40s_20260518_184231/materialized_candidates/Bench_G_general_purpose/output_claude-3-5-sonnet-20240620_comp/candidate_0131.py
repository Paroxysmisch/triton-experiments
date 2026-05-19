import torch
import triton
import triton.language as tl

# Define block sizes for efficient memory access and computation
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Optional activation
    ACTIVATION: tl.constexpr,
    # Block sizes (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = A @ B with optional leaky ReLU activation
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of blocks in grid
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Block ID for current program
    block_id = pid
    block_m = block_id // grid_n
    block_n = block_id % grid_n

    # Starting indices for this block
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Create mask for bounds checking
        k_mask = k + offs_k < K
        
        # Load blocks from A and B
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                   mask=offs_m[:, None] < M & k_mask[None, :])
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                   mask=k_mask[:, None] & (offs_n[None, :] < N))
        
        # Compute matrix multiplication for this block
        accumulator += tl.dot(a, b)
    
    # Apply activation if requested
    if ACTIVATION:
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)
    
    # Store result
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             accumulator, mask=m_mask & n_mask)

def matmul(a: torch.Tensor, b: torch.Tensor, activation: bool = False) -> torch.Tensor:
    """
    Compute C = A @ B with optional leaky ReLU activation
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        activation: Whether to apply leaky ReLU activation
    
    Returns:
        Output tensor of shape (M, N)
    """
    # Check input dimensions
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Extract dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
