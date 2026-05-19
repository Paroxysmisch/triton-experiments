import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_an, stride_bm, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel for matrix multiplication of (M x K) * (K x N) = (M x N),
    handling strides for potential batched inputs.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # For loop over K dimension in BLOCK_SIZE_K steps
    for k_iter in range(0, K, BLOCK_SIZE_K):
        rk = k_iter + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_an,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0
        )
        b = tl.load(
            B_ptr + rk[:, None] * stride_bm + rn[None, :] * stride_bn,
            mask=(rk[:, None] < K) & (rn[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a.astype(tl.float32), b.astype(tl.float32))
    
    # Store result
    c = acc
    # Write back to global memory
    tl.store(
        C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        c,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

def _matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Utility function to dispatch Triton matmul kernel.
    A: (M, K), B: (K, N) -> (M, N)
    """
    # Shapes
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible dimensions for matmul"

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Strides
    stride_am = A.stride(0)
    stride_an = A.stride(1)
    stride_bm = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Grid
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (
        ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M ),
        ( (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )
    )

    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_an, stride_bm, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C

def matrix_power_eig(A: torch.Tensor, k: complex, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    def matrix_power_eig(A, k, *, out=None) -> Tensor
    
    Computes the matrix power A^k of a square matrix A (or batch of matrices) using eigendecomposition.
    A must be diagonalizable for accurate results. The result may be complex even if A is real.
    
    A^k = V diag(Λ^k) V^{-1},
    where A = V diag(Λ) V^{-1}.
    
    Args:
        A (Tensor): Input tensor of shape (*, n, n).
        k (float or complex): The exponent.
    Keyword args:
        out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    """
    # Check if A is at least 2D and square
    if A.dim() < 2:
        raise ValueError("Input tensor A must have at least 2 dimensions.")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("Last two dimensions of A must be square matrices.")

    # Perform batch processing
    # (We will handle each matrix in the batch individually for simplicity)
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    # Flatten batch dims for processing
    A_flat = A.reshape(batch_size, n, n)
    
    # Prepare output
    if out is not None:
        if out.shape != A.shape:
            raise ValueError("out must have the same shape as A.")
        out_flat = out.reshape(batch_size, n, n)
    else:
        out_flat = None

    results = []
    for i in range(batch_size):
        # Extract single matrix
        Ai = A_flat[i]
        
        # Compute eigen decomposition (may return complex even for real input)
        eigvals, eigvecs = torch.linalg.eig(Ai)
        # Power the eigenvalues
        eigvals_pow = eigvals**k
        
        # Construct diag(Λ^k)
        D_pow = torch.diag(eigvals_pow)
        
        # We need V diag(Λ^k) V^(-1)
        # 1) M1 = V * diag(Λ^k)  (use Triton matmul for speed)
        M1 = _matmul_triton(eigvecs, D_pow)
        
        # 2) inverse of V
        Vinv = torch.linalg.inv(eigvecs)
        
        # 3) out = M1 * Vinv
        Ai_k = _matmul_triton(M1, Vinv)
        
        # Cast to input dtype if needed
        if Ai_k.dtype != A.dtype:
            # Handle complex separately if A is complex
            if A.dtype.is_complex or Ai_k.is_complex():
                Ai_k = Ai_k.to(torch.complex64 if A.dtype in [torch.complex64, torch.complex128] else A.dtype)
            else:
                Ai_k = Ai_k.to(A.dtype)
        
        results.append(Ai_k)

    # Stack back into batch shape
    results_tensor = torch.stack(results, dim=0).reshape(*batch_dims, n, n)

    if out_flat is not None:
        out_flat.copy_(results_tensor)
        return out
    else:
        return results_tensor
