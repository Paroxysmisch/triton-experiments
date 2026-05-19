import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return tl.where(x > 0, x, alpha * x)

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
    # Activation function name
    activation_type: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = activation(A @ B) using block-level matrix multiplication
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of blocks in N dimension
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Block indices
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n
    
    # Starting indices for this block
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                   mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K))
        b = tl.load(b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                   mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N))
        
        # Compute block-level matrix multiplication
        acc += tl.dot(a, b)
    
    # Apply activation function if specified
    if activation_type == 1:  # leaky ReLU
        acc = leaky_relu(acc)
    
    # Store results
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=mask)

def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = None):
    """
    Compute C = activation(A @ B)
    
    Parameters:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        activation: Activation function to apply ('leaky_relu' or None)
    
    Returns:
        c: Output tensor of shape (M, N)
    """
    # Check input dimensions
    assert len(a.shape) == len(b.shape) == 2
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Extract dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Compute grid size
    grid = lambda meta: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    # Determine activation type
    activation_type = 1 if activation == 'leaky_relu' else 0
    
    # Launch kernel
    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        activation_type=activation_type,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
