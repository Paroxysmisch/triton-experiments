import triton
import triton.language as tl
import torch

# Triton kernel to perform LU decomposition with partial pivoting
@triton.jit
def lu_kernel(
    X,  # Pointer to the input matrix
    P,  # Pointer to the permutation matrix
    L,  # Pointer to the lower triangular matrix
    U,  # Pointer to the upper triangular matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix,
    diag_offset: tl.constexpr,  # Diagonal offset
    M_BLOCK_SIZE: tl.constexpr,  # Block size for rows
    N_BLOCK_SIZE: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    col = pid * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE)[None, :]
    
    # Initialize L and U
    l_mask = row >= col
    u_mask = row <= col
    tl.store(L + row * N + col, tl.select(l_mask, X[row * N + col], 0.0))
    tl.store(U + row * N + col, tl.select(u_mask, X[row * N + col], 0.0))

    # Apply partial pivoting
    max_val = tl.max(tl.where(col == row, X[row * N + col], -tl.inf))
    p_row = tl.argmax(max_val)
    tl.atomic_max(P + row * N + p_row, 1.0)
    tl.atomic_min(P + row * N + p_row, 0.0)
    tl.atomic_add(P + row * N + p_row, row * N + col)

    # Update U and L after pivoting
    u_mask = row <= col
    tl.store(U + row * N + col, tl.select(u_mask, X[row * N + col], 0.0))
    l_mask = row >= col
    tl.store(L + row * N + col, tl.select(l_mask, X[row * N + col], 0.0))

    # Perform Gaussian elimination
    for k in range(N):
        pivot_col = k * N + k
        pivot_val = tl.load(U + pivot_col)
        for i in range(k + 1, M):
            i_col = i * N + k
            factor = tl.load(U + i_col) / pivot_val
            tl.store(U + i_col, factor)
            for j in range(k + 1, N):
                j_col = i * N + j
                update_val = tl.load(U + j_col) - factor * tl.load(U + pivot_col + j - k)
                tl.store(U + j_col, update_val)
