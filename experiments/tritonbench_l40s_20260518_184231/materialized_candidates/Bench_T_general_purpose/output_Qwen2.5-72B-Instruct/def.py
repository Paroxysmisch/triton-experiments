import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A_ptr,  # Pointer to the input matrix A
    P_ptr,  # Pointer to the permutation matrix P
    L_ptr,  # Pointer to the lower triangular matrix L
    U_ptr,  # Pointer to the upper triangular matrix U
    n,  # Size of the matrix
    pivot,  # Whether to use pivoting
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE

    # Load the matrix A into shared memory
    A = tl.load(A_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    P = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.int32)
    L = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    U = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for i in range(BLOCK_SIZE):
        if pivot:
            # Find the pivot row
            max_val = tl.max(A[i:, i], axis=0)
            pivot_row = tl.argmax(A[i:, i], axis=0) + i
            # Swap rows in A, P, L, and U
            A[[i, pivot_row]] = A[[pivot_row, i]]
            P[[i, pivot_row]] = P[[pivot_row, i]]
            L[[i, pivot_row]] = L[[pivot_row, i]]
            U[[i, pivot_row]] = U[[pivot_row, i]]

        # Compute the L and U matrices
        U[i, i:] = A[i, i:]
        L[i+1:, i] = A[i+1:, i] / U[i, i]

        # Update the remaining part of A
        A[i+1:, i+1:] -= tl.outer(L[i+1:, i], U[i, i+1:])

    # Store the results back to global memory
    tl.store(L_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), L, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)
    tl.store(U_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), U, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)
    tl.store(P_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), P, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)

@triton.jit
def solve_linear_systems_kernel(
    L_ptr,  # Pointer to the lower triangular matrix L
    U_ptr,  # Pointer to the upper triangular matrix U
    P_ptr,  # Pointer to the permutation matrix P
    B_ptr,  # Pointer to the right-hand side matrix B
    X_ptr,  # Pointer to the output matrix X
    n,  # Size of the matrix
    k,  # Number of right-hand sides
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE

    # Load the matrices into shared memory
    L = tl.load(L_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    U = tl.load(U_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    P = tl.load(P_ptr + block_start * n + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    B = tl.load(B_ptr + block_start * k + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < n, other=0.0)
    X = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)

    # Apply permutation
    B = tl.dot(P.T, B)

    # Solve L y = B
    for i in range(BLOCK_SIZE):
        X[i] = (B[i] - tl.dot(L[i, :i], X[:i])) / L[i, i]

    # Solve U x = y
    for i in range(BLOCK_SIZE - 1, -1, -1):
        X[i] = (X[i] - tl.dot(U[i, i+1:], X[i+1:])) / U[i, i]

    # Store the results back to global memory
    tl.store(X_ptr + block_start * k + tl.arange(0, BLOCK_SIZE), X, mask=block_start + tl.arange(0, BLOCK_SIZE) < n)

import torch
import triton

def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    # Get the shape of the input tensors
    *batch_shape, n, _ = A.shape
    *_, n, k = Bs.shape

    # Flatten the batch dimensions
    A = A.view(-1, n, n)
    Bs = Bs.view(-1, n, k)

    # Allocate memory for the permutation matrix P, L, and U
    P = torch.eye(n, device=A.device, dtype=torch.int32).repeat(A.size(0), 1, 1)
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)

    # Perform the LU decomposition
    grid = (A.size(0),)
    lu_decomposition_kernel[grid](
        A, P, L, U, n, pivot, BLOCK_SIZE=32
    )

    # Allocate memory for the output tensor if not provided
    if out is None:
        out = torch.zeros_like(Bs)

    # Solve the linear systems
    solve_linear_systems_kernel[grid](
        L, U, P, Bs, out, n, k, BLOCK_SIZE=32
    )

    # Reshape the output tensor back to the original batch shape
    out = out.view(*batch_shape, n, k)

    return out
