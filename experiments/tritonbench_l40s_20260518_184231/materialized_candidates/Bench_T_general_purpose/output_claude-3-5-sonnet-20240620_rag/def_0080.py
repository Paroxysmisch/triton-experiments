import torch
import triton
import triton.language as tl
import math

@triton.jit
def qr_solve_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, x_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block indices
    block_m = pid // (N // BLOCK_SIZE_N)
    block_n = pid % (N // BLOCK_SIZE_N)
    
    # Compute offsets
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulators for Q and R
    q = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    r = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load matrix A block
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    
    # QR decomposition using modified Gram-Schmidt
    for i in range(BLOCK_SIZE_N):
        # Get column vector
        v = a[:, i]
        r_ii = tl.sqrt(tl.sum(v * v))
        q[:, i] = v / r_ii
        r[i, i] = r_ii
        
        # Update remaining columns
        for j in range(i + 1, BLOCK_SIZE_N):
            r[i, j] = tl.sum(q[:, i] * a[:, j])
            a[:, j] = a[:, j] - r[i, j] * q[:, i]
    
    # Load b block
    b = tl.load(b_ptr + offs_m[:, None] * K + tl.arange(0, K)[None, :])
    
    # Compute Q^T * b
    qtb = tl.zeros([BLOCK_SIZE_N, K], dtype=tl.float32)
    for i in range(BLOCK_SIZE_N):
        qtb[i, :] = tl.sum(q[:, i:i+1] * b, axis=0)
    
    # Solve R * x = Q^T * b using back substitution
    x = tl.zeros([BLOCK_SIZE_N, K], dtype=tl.float32)
    for i in range(BLOCK_SIZE_N-1, -1, -1):
        x[i, :] = (qtb[i, :] - tl.sum(r[i, i+1:] * x[i+1:, :], axis=0)) / r[i, i]
    
    # Store result
    mask_out = offs_n[:, None] < N
    tl.store(x_ptr + offs_n[:, None] * K + tl.arange(0, K)[None, :], x, mask=mask_out)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the linear system Ax = b using QR decomposition.
    
    Args:
        A: Input matrix of shape (m, n) where m >= n
        b: Right-hand side tensor of shape (m, k)
    
    Returns:
        x: Solution tensor of shape (n, k)
    """
    assert A.dim() == 2 and b.dim() == 2, "Input tensors must be 2-dimensional"
    M, N = A.shape
    assert M >= N, "Number of rows must be greater than or equal to number of columns"
    assert b.shape[0] == M, "Incompatible dimensions between A and b"
    K = b.shape[1]
    
    # Determine block sizes
    BLOCK_SIZE_M = triton.next_power_of_2(min(M, 32))
    BLOCK_SIZE_N = triton.next_power_of_2(min(N, 32))
    BLOCK_SIZE_K = triton.next_power_of_2(min(K, 32))
    
    # Create output tensor
    x = torch.empty((N, K), device=A.device, dtype=A.dtype)
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    qr_solve_kernel[grid](
        A, b, x,
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return x
