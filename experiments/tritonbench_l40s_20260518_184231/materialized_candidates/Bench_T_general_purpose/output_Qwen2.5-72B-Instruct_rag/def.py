import triton
import triton.language as tl
import torch

# Triton kernel to perform LU decomposition and solve multiple linear systems
@triton.jit
def solve_multiple_lu_kernel(
    A,  # Pointer to the coefficient matrix
    Bs,  # Pointer to the right-hand side tensor
    X,  # Pointer to the output tensor
    batch,  # Number of batch elements
    n,  # Size of the square matrix
    k,  # Number of right-hand sides
    pivot,  # Whether to use pivoting
    BLOCK_SIZE: tl.constexpr,  # Block size for matrix operations
):
    pid = tl.program_id(0)
    batch_id = pid // (n // BLOCK_SIZE)
    row_id = (pid % (n // BLOCK_SIZE)) * BLOCK_SIZE

    # Load the matrix A and Bs for the current batch
    A_batch = A + batch_id * n * n
    Bs_batch = Bs + batch_id * n * k
    X_batch = X + batch_id * n * k

    # Perform LU decomposition
    if pivot:
        P, L, U = lu_decomposition_with_pivoting(A_batch, n, BLOCK_SIZE)
    else:
        L, U = lu_decomposition_without_pivoting(A_batch, n, BLOCK_SIZE)
        P = None

    # Solve for each right-hand side
    for i in range(k):
        b = Bs_batch + i * n
        x = X_batch + i * n

        # Apply permutation if pivoting is used
        if pivot:
            b_permuted = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for j in range(n // BLOCK_SIZE):
                b_permuted[j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] = P[j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE] @ b[j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
            b = b_permuted

        # Solve L y = b
        y = forward_substitution(L, b, n, BLOCK_SIZE)

        # Solve U x = y
        x = backward_substitution(U, y, n, BLOCK_SIZE)

        # Store the result
        tl.store(x, X_batch + i * n)

# Helper function to perform LU decomposition with pivoting
@triton.jit
def lu_decomposition_with_pivoting(A, n, BLOCK_SIZE):
    P = tl.eye(n, dtype=tl.int32)
    L = tl.zeros((n, n), dtype=tl.float32)
    U = A.clone()

    for i in range(n):
        max_val = -1e9
        max_row = i
        for j in range(i, n):
            if U[j, i] > max_val:
                max_val = U[j, i]
                max_row = j

        # Swap rows in U and P
        U[[i, max_row], :] = U[[max_row, i], :]
        P[[i, max_row], :] = P[[max_row, i], :]

        # Update L
        L[i, i] = 1.0
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:n] -= L[j, i] * U[i, i:n]

    return P, L, U

# Helper function to perform LU decomposition without pivoting
@triton.jit
def lu_decomposition_without_pivoting(A, n, BLOCK_SIZE):
    L = tl.zeros((n, n), dtype=tl.float32)
    U = A.clone()

    for i in range(n):
        L[i, i] = 1.0
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:n] -= L[j, i] * U[i, i:n]

    return L, U

# Helper function to perform forward substitution
@triton.jit
def forward_substitution(L, b, n, BLOCK_SIZE):
    y = tl.zeros((n,), dtype=tl.float32)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    return y

# Helper function to perform backward substitution
@triton.jit
def backward_substitution(U, y, n, BLOCK_SIZE):
    x = tl.zeros((n,), dtype=tl.float32)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x

# Python wrapper function
def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    A = A.contiguous()
    Bs = Bs.contiguous()
    batch = A.shape[:-2]
    n = A.shape[-1]
    k = Bs.shape[-1]

    if out is None:
        out = torch.empty_like(Bs)

    assert A.shape[-2] == n, "A must be a square matrix"
    assert Bs.shape[-2] == n, "Bs must have the same number of rows as A"
    assert A.shape[:-2] == Bs.shape[:-2], "A and Bs must have the same batch dimensions"

    with torch.cuda.device(A.device):
        grid = lambda meta: (triton.cdiv(batch.numel() * n, meta["BLOCK_SIZE"]),)
        solve_multiple_lu_kernel[grid](A, Bs, out, batch.numel(), n, k, pivot, BLOCK_SIZE=32)

    return out
