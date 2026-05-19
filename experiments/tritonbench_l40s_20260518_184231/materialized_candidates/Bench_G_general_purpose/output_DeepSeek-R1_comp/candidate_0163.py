import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride values (in elements)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Activation name
    activation: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Determine block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create ranges for block dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        k_remaining = K - k
        k_effective = min(BLOCK_SIZE_K, k_remaining)
        
        # Load A block with masking for boundary conditions
        a_ptrs = a_ptr + (rm[:, None] * stride_am) + ((k + rk[None, :]) * stride_ak)
        a_mask = (rm[:, None] < M) & (rk[None, :] < k_effective)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with masking for boundary conditions
        b_ptrs = b_ptr + ((k + rk[:, None]) * stride_bk) + (rn[None, :] * stride_bn)
        b_mask = (rk[:, None] < k_effective) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute partial matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Apply activation function
    if activation == "leaky_relu":
        acc = tl.where(acc >= 0, acc, 0.01 * acc)
    elif activation == "relu":
        acc = tl.maximum(acc, 0.0)
    
    # Store result with masking for boundary conditions
    c_ptrs = c_ptr + (rm[:, None] * stride_cm) + (rn[None, :] * stride_cn)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul(a, b, activation=None, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
    # Validate dimensions
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure grid
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        activation=activation or "none",
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c

# Example usage
a = torch.randn(1024, 512, device='cuda')
b = torch.randn(512, 2048, device='cuda')

# Basic matrix multiplication
c = matmul(a, b)

# With ReLU activation
c_relu = matmul(a, b, activation="relu")

# With Leaky ReLU activation
c_lrelu = matmul(a, b, activation="leaky_relu")
