import torch
import triton
import triton.language as tl

@triton.jit
def eig_kernel(A, V, L, n, m):
    # Triton kernel for computing eigenvalues and eigenvectors of a square matrix
    idx = tl.program_id(0)
    if idx < m:
        # Compute the QR decomposition of A
        Q, R, k = triton_linalg.qr(A[idx], max_iter=100)
        # Iteratively refine the QR decomposition
        for _ in range(k):
            Q, R, _ = triton_linalg.qr(A[idx], max_iter=100)
            A[idx] = triton_linalg.matmul(R, Q)
        # Compute the eigenvalues and eigenvectors
        triton_linalg.eig_vec(A[idx], V[idx], L[idx], n)

def eig(A, *, out=None):
    # Function to compute eigenvalues and eigenvectors of a square matrix
    A = torch.as_tensor(A)
    if A.dim() < 2:
        raise RuntimeError("A must be at least 2D")
    if A.size(-1) != A.size(-2):
        raise RuntimeError("A must be a square matrix")
    n = A.size(-1)
    m = A.numel() // n ** 2
    A_reshaped = A.reshape(m, n, n)
    if A.is_cuda:
        torch.cuda.synchronize()
    # Allocate output tensors
    L = torch.empty(m, n, dtype=A.dtype, device=A.device)
    V = torch.empty(m, n, n, dtype=A.dtype, device=A.device)
    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(1, meta["BLOCK_SIZE"]),)
    eig_kernel[grid](A_reshaped, V, L, n, m)
    return L.reshape(A.shape[:-1] + (n,)), V.reshape(A.shape[:-1] + (n, n))
