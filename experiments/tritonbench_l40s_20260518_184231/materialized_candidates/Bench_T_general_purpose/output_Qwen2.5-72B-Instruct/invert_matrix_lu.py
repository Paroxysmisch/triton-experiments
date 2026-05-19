import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A, P, L, U, n, m, batch_size, pivot, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    if bid >= batch_size:
        return

    # Compute the row and column indices
    row = pid // m
    col = pid % m

    # Load the matrix A
    A_ptr = A + bid * n * m + row * m + col
    P_ptr = P + bid * n + row
    L_ptr = L + bid * n * m + row * m + col
    U_ptr = U + bid * n * m + row * m + col

    # Initialize P, L, and U
    tl.store(P_ptr, row)
    tl.store(L_ptr, 0.0)
    tl.store(U_ptr, 0.0)

    for k in range(n):
        # Compute U
        if row == k:
            for j in range(k, m):
                U_ptr[j] = A_ptr[j]

        # Compute L
        if col == k:
            for i in range(k + 1, n):
                L_ptr[i * m + k] = A_ptr[i * m + k] / U_ptr[k * m + k]

        # Update A
        for i in range(k + 1, n):
            for j in range(k + 1, m):
                A_ptr[i * m + j] -= L_ptr[i * m + k] * U_ptr[k * m + j]

        # Pivoting
        if pivot:
            max_val = -1.0
            max_row = k
            for i in range(k, n):
                if tl.abs(U_ptr[i * m + k]) > max_val:
                    max_val = tl.abs(U_ptr[i * m + k])
                    max_row = i

            if max_row != k:
                # Swap rows in A, L, U, and P
                for j in range(k, m):
                    A_ptr[k * m + j], A_ptr[max_row * m + j] = A_ptr[max_row * m + j], A_ptr[k * m + j]
                    U_ptr[k * m + j], U_ptr[max_row * m + j] = U_ptr[max_row * m + j], U_ptr[k * m + j]
                for j in range(k + 1, n):
                    L_ptr[j * m + k], L_ptr[j * m + max_row] = L_ptr[j * m + max_row], L_ptr[j * m + k]
                P_ptr[k], P_ptr[max_row] = P_ptr[max_row], P_ptr[k]

@triton.jit
def solve_triangular_kernel(
    A, B, n, m, batch_size, lower, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    if bid >= batch_size:
        return

    # Compute the row and column indices
    row = pid // m
    col = pid % m

    # Load the matrix A and vector B
    A_ptr = A + bid * n * m + row * m + col
    B_ptr = B + bid * n + row

    for k in range(n):
        if lower:
            if row == k:
                B_ptr[k] /= A_ptr[k * m + k]
            if row > k:
                B_ptr[row] -= A_ptr[row * m + k] * B_ptr[k]
        else:
            if row == n - k - 1:
                B_ptr[row] /= A_ptr[row * m + row]
            if row < n - k - 1:
                B_ptr[row] -= A_ptr[row * m + (n - k - 1)] * B_ptr[n - k - 1]

@triton.jit
def invert_matrix_kernel(
    A, P, L, U, A_inv, n, m, batch_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    if bid >= batch_size:
        return

    # Compute the row and column indices
    row = pid // m
    col = pid % m

    # Load the matrix A, P, L, U, and A_inv
    A_ptr = A + bid * n * m + row * m + col
    P_ptr = P + bid * n + row
    L_ptr = L + bid * n * m + row * m + col
    U_ptr = U + bid * n * m + row * m + col
    A_inv_ptr = A_inv + bid * n * m + row * m + col

    # Initialize A_inv to the identity matrix
    if row == col:
        tl.store(A_inv_ptr, 1.0)
    else:
        tl.store(A_inv_ptr, 0.0)

    # Solve Y = L^{-1} P
    for i in range(n):
        for j in range(n):
            if i == j:
                A_inv_ptr[i * m + j] = 1.0
            else:
                A_inv_ptr[i * m + j] = 0.0

    solve_triangular_kernel(L, A_inv, n, m, batch_size, True, BLOCK_SIZE)

    # Apply permutation P
    for i in range(n):
        for j in range(n):
            temp = A_inv_ptr[i * m + j]
            A_inv_ptr[i * m + j] = A_inv_ptr[P_ptr[i] * m + j]
            A_inv_ptr[P_ptr[i] * m + j] = temp

    # Solve A^{-1} = U^{-1} Y
    solve_triangular_kernel(U, A_inv, n, m, batch_size, False, BLOCK_SIZE)

import torch
import triton
import triton.language as tl

def invert_matrix_lu(A, *, pivot=True, out=None):
    # Get the shape of the input tensor
    batch_size, n, m = A.shape

    # Check if the input is a square matrix
    assert n == m, "Input matrix must be square"

    # Create the permutation matrix P
    P = torch.arange(n, device=A.device).repeat(batch_size, 1)

    # Create the L and U matrices
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)

    # Perform LU decomposition
    grid = (n * m, batch_size)
    lu_decomposition_kernel[grid](A, P, L, U, n, m, batch_size, pivot, BLOCK_SIZE=16)

    # Initialize the inverse matrix
    if out is None:
        A_inv = torch.zeros_like(A)
    else:
        A_inv = out

    # Compute the inverse
    invert_matrix_kernel[grid](A, P, L, U, A_inv, n, m, batch_size, BLOCK_SIZE=16)

    return A_inv
