import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, N, K,
    # Stride variables
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    alpha: tl.constexpr = 0.01
):
    """Kernel for matrix multiplication C = A @ B with optional activation."""
    # Determine block indices
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    
    # Create block offsets
    offs_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(num_blocks_k):
        k_offset = k * BLOCK_SIZE_K
        
        # Load A block with boundary checks
        a_ptrs = A + (offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with boundary checks
        b_ptrs = B + ((k_offset + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = ((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute partial matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        acc = tl.where(acc >= 0, acc, acc * alpha)
    
    # Write back result with boundary checks
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = None, alpha: float = 0.01):
    """
    High-performance matrix multiplication with optional activation.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
        activation: Optional activation ("leaky_relu")
        alpha: Slope for leaky ReLU (default: 0.01)
    
    Returns:
        torch.Tensor: Result matrix C of shape (M, N)
    """
    # Validate inputs
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    assert a.size(1) == b.size(0), f"Dimension mismatch: {a.shape} vs {b.shape}"
    
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configure grid and kernel parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
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
        ACTIVATION=activation,
        alpha=alpha,
        num_warps=4,
        num_stages=3
    )
    
    return c

# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    M, K, N = 1024, 512, 2048
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Compute without activation
    c_triton = matmul(a, b)
    
    # Compute with leaky ReLU
    c_activated = matmul(a, b, activation="leaky_relu", alpha=0.05)
    
    # Verify against PyTorch implementation
    c_ref = torch.nn.functional.leaky_relu(a @ b.to(torch.float32), 0.05).half()
    print(f"Activation max error: {torch.max(torch.abs(c_activated - c_ref))}")
