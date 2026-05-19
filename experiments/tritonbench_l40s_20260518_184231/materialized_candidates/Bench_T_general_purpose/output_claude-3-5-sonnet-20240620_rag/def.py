import triton
import triton.language as tl
import torch

# Triton kernel for LU decomposition
@triton.jit
def lu_decomposition_kernel(A, L, U, P, n, block_size: tl.constexpr):
    # Compute LU decomposition with partial pivoting
    row = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = row < n
    for k in range(n):
        # Pivoting
        pivot = tl.argmax(A[k:n, k], axis=0) + k
        P[k], P[pivot] = P[pivot], P[k]
        A[k:n, k], A[pivot, k] = A[pivot, k], A[k:n, k]
        
        # LU decomposition
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / A[k, k]
            A[i, k:n] -= L[i, k] * A[k, k:n]

    # Store L and U
    L += tl.tril(A, diagonal=-1) + tl.eye(n)
    U += tl.triu(A)

# Triton kernel to solve linear systems
@triton.jit
def solve_linear_system_kernel(L, U, P, B, X, n, k, block_size: tl.constexpr):
    row = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = row < n
    for i in range(k):
        # Apply permutation
        b_permuted = B[row] if P[row] else B[P[row]]
        
        # Solve L y = b
        y = tl.zeros_like(b_permuted)
        for j in range(n):
            y[j] = b_permuted[j] - tl.sum(L[j, :j] * y[:j])
        
        # Solve U x = y
        for j in range(n - 1, -1, -1):
            X[row, i] = (y[j] - tl.sum(U[j, j + 1:] * X[row, i][j + 1:])) / U[j, j]

# Wrapper function
def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    A = A.contiguous()
    Bs = Bs.contiguous()
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    # Initialize L, U, and P
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)
    P = torch.arange(n, device=A.device)

    # LU decomposition
    grid = lambda meta: (triton.cdiv(n, meta['block_size']),)
    lu_decomposition_kernel[grid](A, L, U, P, n, block_size=32)

    # Solve for each right-hand side
    if out is None:
        out = torch.empty_like(Bs)
    
    grid = lambda meta: (triton.cdiv(n, meta['block_size']),)
    solve_linear_system_kernel[grid](L, U, P, Bs, out, n, k, block_size=32)

    return out
