import triton
import triton.language as tl

# Triton kernel for LU decomposition
@triton.jit
def lu_decomposition_kernel(A, P, L, U, pivot, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_size = A.shape[0]
    n = A.shape[1]
    batch_idx = pid // n
    row_idx = pid % n

    if row_idx < n:
        for col_idx in range(n):
            if pivot:
                # Partial pivoting
                max_idx = tl.argmax(tl.abs(A[batch_idx, row_idx + tl.arange(0, BLOCK_SIZE), col_idx]), axis=0)
                A[batch_idx, row_idx, col_idx], A[batch_idx, row_idx + max_idx, col_idx] = \
                    A[batch_idx, row_idx + max_idx, col_idx], A[batch_idx, row_idx, col_idx]
                P[batch_idx, row_idx, col_idx], P[batch_idx, row_idx + max_idx, col_idx] = \
                    P[batch_idx, row_idx + max_idx, col_idx], P[batch_idx, row_idx, col_idx]

            if row_idx == col_idx:
                L[batch_idx, row_idx, col_idx] = 1.0
                U[batch_idx, row_idx, col_idx] = A[batch_idx, row_idx, col_idx]
            else:
                if row_idx < col_idx:
                    L[batch_idx, row_idx, col_idx] = A[batch_idx, row_idx, col_idx] / U[batch_idx, col_idx, col_idx]
                else:
                    U[batch_idx, row_idx, col_idx] = A[batch_idx, row_idx, col_idx] - \
                        tl.sum(L[batch_idx, row_idx, :col_idx] * U[batch_idx, :col_idx, col_idx])

# Triton kernel for solving L * Y = P
@triton.jit
def solve_ly_kernel(L, P, Y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_size = L.shape[0]
    n = L.shape[1]
    batch_idx = pid // n
    row_idx = pid % n

    if row_idx < n:
        for col_idx in range(n):
            Y[batch_idx, row_idx, col_idx] = P[batch_idx, row_idx, col_idx]
            for k in range(col_idx):
                Y[batch_idx, row_idx, col_idx] -= L[batch_idx, row_idx, k] * Y[batch_idx, k, col_idx]

# Triton kernel for solving U * A^{-1} = Y
@triton.jit
def solve_uinv_kernel(U, Y, A_inv, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_size = U.shape[0]
    n = U.shape[1]
    batch_idx = pid // n
    row_idx = pid % n

    if row_idx < n:
        for col_idx in range(n - 1, -1, -1):
            A_inv[batch_idx, row_idx, col_idx] = Y[batch_idx, row_idx, col_idx]
            for k in range(col_idx + 1, n):
                A_inv[batch_idx, row_idx, col_idx] -= U[batch_idx, row_idx, k] * A_inv[batch_idx, k, col_idx]
            A_inv[batch_idx, row_idx, col_idx] /= U[batch_idx, col_idx, col_idx]

# Wrapper function
def invert_matrix_lu(A, *, pivot=True, out=None):
    batch_size, n, _ = A.shape
    if out is None:
        out = tl.zeros_like(A)
    
    # Initialize P, L, U
    P = tl.eye(n, dtype=A.dtype)
    L = tl.zeros((batch_size, n, n), dtype=A.dtype)
    U = tl.zeros((batch_size, n, n), dtype=A.dtype)
    
    # Perform LU decomposition
    lu_decomposition_kernel[(batch_size * n,)](A, P, L, U, pivot, BLOCK_SIZE=n)
    
    # Solve L * Y = P
    Y = tl.zeros((batch_size, n, n), dtype=A.dtype)
    solve_ly_kernel[(batch_size * n,)](L, P, Y, BLOCK_SIZE=n)
    
    # Solve U * A^{-1} = Y
    solve_uinv_kernel[(batch_size * n,)](U, Y, out, BLOCK_SIZE=n)
    
    return out
