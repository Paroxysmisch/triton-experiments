import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(A, P, L, U, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    for k in range(n):
        # Compute pivot
        pivot = tl.argmax(tl.abs(A[k, k:n]), axis=0) + k
        P[k, k] = 1.0
        if pivot != k:
            P[k, k] = 0.0
            P[pivot, pivot] = 0.0
            P[k, pivot] = 1.0
            P[pivot, k] = 1.0
            A[[k, pivot], :] = A[[pivot, k], :]

        # Compute L and U
        for j in range(k + 1, n):
            L[j, k] = A[j, k] / A[k, k]
            A[j, k:n] = A[j, k:n] - L[j, k] * A[k, k:n]

    for i in range(n):
        U[i, i:n] = A[i, i:n]

@triton.jit
def forward_substitution_kernel(L, P, b, y, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] = y[i] - L[i, j] * y[j]
        y[i] = y[i] / L[i, i]

@triton.jit
def backward_substitution_kernel(U, y, x, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] = x[i] - U[i, j] * x[j]
        x[i] = x[i] / U[i, i]

import torch
import triton
import triton.language as tl

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix A must be square"
    assert b.shape == (n,), "Tensor b must have shape (n,)"

    # Initialize tensors for P, L, U, y, and x
    P = torch.eye(n, device=A.device, dtype=A.dtype)
    L = torch.zeros((n, n), device=A.device, dtype=A.dtype)
    U = torch.zeros((n, n), device=A.device, dtype=A.dtype)
    y = torch.zeros(n, device=A.device, dtype=A.dtype)
    x = torch.zeros(n, device=A.device, dtype=A.dtype)

    # Perform LU decomposition
    grid = (triton.cdiv(n, 32),)
    lu_decomposition_kernel[grid](A, P, L, U, n, BLOCK_SIZE=32)

    # Solve L @ y = P @ b
    forward_substitution_kernel[grid](L, P, b, y, n, BLOCK_SIZE=32)

    # Solve U @ x = y
    backward_substitution_kernel[grid](U, y, x, n, BLOCK_SIZE=32)

    return x
