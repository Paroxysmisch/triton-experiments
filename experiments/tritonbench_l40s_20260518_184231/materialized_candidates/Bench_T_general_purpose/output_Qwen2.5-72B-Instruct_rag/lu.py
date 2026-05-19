import triton
import triton.language as tl
import torch

# Triton kernel for LU decomposition with partial pivoting
@triton.jit
def lu_kernel_pivoting(
    A,  # Pointer to the input matrix
    P,  # Pointer to the permutation matrix
    L,  # Pointer to the lower triangular matrix
    U,  # Pointer to the upper triangular matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    M_BLOCK_SIZE: tl.constexpr,  # Block size for rows
    N_BLOCK_SIZE: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    A += row * N
    P += row * N
    L += row * N
    U += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        # Load the current block of A
        a = tl.load(A + cols, mask, other=0.0)

        # Compute the pivot and swap rows
        pivot_row = tl.argmax(tl.abs(a), axis=0)
        pivot_row = tl.where(pivot_row < M, pivot_row, 0)
        pivot_row = tl.where(pivot_row < N, pivot_row, 0)

        # Swap rows in A, P, L, and U
        temp = tl.load(A + pivot_row * N + cols, mask, other=0.0)
        tl.store(A + pivot_row * N + cols, a, mask=mask)
        tl.store(A + row * N + cols, temp, mask=mask)

        temp = tl.load(P + pivot_row * N + cols, mask, other=0.0)
        tl.store(P + pivot_row * N + cols, a, mask=mask)
        tl.store(P + row * N + cols, temp, mask=mask)

        temp = tl.load(L + pivot_row * N + cols, mask, other=0.0)
        tl.store(L + pivot_row * N + cols, a, mask=mask)
        tl.store(L + row * N + cols, temp, mask=mask)

        temp = tl.load(U + pivot_row * N + cols, mask, other=0.0)
        tl.store(U + pivot_row * N + cols, a, mask=mask)
        tl.store(U + row * N + cols, temp, mask=mask)

        # Compute the L and U factors
        for i in range(row + 1, M):
            l = tl.load(A + i * N + cols, mask, other=0.0)
            u = tl.load(A + row * N + cols, mask, other=0.0)
            l = l / u
            tl.store(L + i * N + cols, l, mask=mask)
            a = a - l * u
            tl.store(A + i * N + cols, a, mask=mask)

# Triton kernel for LU decomposition without pivoting
@triton.jit
def lu_kernel_no_pivoting(
    A,  # Pointer to the input matrix
    L,  # Pointer to the lower triangular matrix
    U,  # Pointer to the upper triangular matrix
    M,  # Number of rows in the input matrix
    N,  # Number of columns in the input matrix
    M_BLOCK_SIZE: tl.constexpr,  # Block size for rows
    N_BLOCK_SIZE: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)[:, None]
    m_mask = row < M
    A += row * N
    L += row * N
    U += row * N

    for n_offset in range(0, N, N_BLOCK_SIZE):
        cols = n_offset + tl.arange(0, N_BLOCK_SIZE)[None, :]
        n_mask = cols < N
        mask = m_mask and n_mask

        # Load the current block of A
        a = tl.load(A + cols, mask, other=0.0)

        # Compute the L and U factors
        for i in range(row + 1, M):
            l = tl.load(A + i * N + cols, mask, other=0.0)
            u = tl.load(A + row * N + cols, mask, other=0.0)
            l = l / u
            tl.store(L + i * N + cols, l, mask=mask)
            a = a - l * u
            tl.store(A + i * N + cols, a, mask=mask)

# Wrapper function for LU decomposition
def lu(A, *, pivot=True, out=None):
    A = A.contiguous()
    M, N = A.shape[-2:]
    batch = A.shape[:-2] if len(A.shape) > 2 else (1,)
    batch_size = int(torch.prod(torch.tensor(batch)))

    if out is None:
        if pivot:
            P = torch.empty(batch + (M, N), dtype=torch.int64, device=A.device)
            L = torch.empty(batch + (M, N), dtype=A.dtype, device=A.device)
            U = torch.empty(batch + (M, N), dtype=A.dtype, device=A.device)
        else:
            P = torch.empty(0, dtype=torch.int64, device=A.device)
            L = torch.empty(batch + (M, N), dtype=A.dtype, device=A.device)
            U = torch.empty(batch + (M, N), dtype=A.dtype, device=A.device)
    else:
        P, L, U = out

    with torch.cuda.device(A.device):
        if pivot:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            lu_kernel_pivoting[grid](A, P, L, U, M, N, M_BLOCK_SIZE=32, N_BLOCK_SIZE=8)
        else:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            lu_kernel_no_pivoting[grid](A, L, U, M, N, M_BLOCK_SIZE=32, N_BLOCK_SIZE=8)

    return P, L, U
