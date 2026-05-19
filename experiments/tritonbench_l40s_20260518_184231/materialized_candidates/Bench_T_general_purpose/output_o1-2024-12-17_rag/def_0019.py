import torch
import triton
import triton.language as tl

@triton.jit
def _lu_factor_kernel(
    A_ptr,  # in-out: A (n x n), overwritten with L and U (L has 1s on diag)
    n,      # size
    k,      # current pivot row
    BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row >= (k + 1)
    # Load pivot (diagonal element)
    pivot = tl.load(A_ptr + k * n + k)
    # Load A[i, k], divide by pivot
    val = tl.load(A_ptr + row * n + k, mask=mask)
    val = val / pivot
    # Store back (L's column k)
    tl.store(A_ptr + row * n + k, val, mask=mask)
    # Update trailing submatrix A[i, j] -= A[i, k] * A[k, j]
    # We'll process columns in a loop for simplicity
    # This loop runs in Python, each call updates a block of columns
    # Each column block iteration can be done in a separate kernel if needed

@triton.jit
def _lu_update_kernel(
    A_ptr,
    n,
    k,
    col_start,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N) + col_start
    row_mask = row >= (k + 1)
    col_mask = col >= (k + 1)
    mask = row_mask[:, None] & col_mask[None, :]
    a_i_k = tl.load(A_ptr + row * n + k, mask=row_mask)
    a_k_j = tl.load(A_ptr + k * n + col, mask=col_mask)
    update = a_i_k[:, None] * a_k_j[None, :]
    old = tl.load(A_ptr + row[:, None] * n + col[None, :], mask=mask)
    new = old - update
    tl.store(A_ptr + row[:, None] * n + col[None, :], new, mask=mask)

@triton.jit
def _forward_sub_kernel(
    A_ptr,
    b_ptr,
    n,
    BLOCK: tl.constexpr
):
    # Each warp works on one row in forward substitution
    i = tl.program_id(0)
    if i < n:
        # load b[i]
        rhs = tl.load(b_ptr + i)
        # sum(L[i, j] * b[j]) for j in [0..i-1]
        j_off = tl.arange(0, BLOCK)
        acc = 0.0
        for start_j in range(0, n, BLOCK):
            j = start_j + j_off
            mask = j < i
            lvals = tl.load(A_ptr + i * n + j, mask=mask)
            bvals = tl.load(b_ptr + j, mask=mask)
            acc += tl.sum(lvals * bvals, where=mask)
        rhs = rhs - acc
        tl.store(b_ptr + i, rhs)

@triton.jit
def _back_sub_kernel(
    A_ptr,
    b_ptr,
    n,
    BLOCK: tl.constexpr
):
    # Each warp works on one row in backward substitution
    i = n - 1 - tl.program_id(0)
    if i >= 0:
        rhs = tl.load(b_ptr + i)
        # sum(U[i, j] * x[j]) for j in [i+1..n-1]
        j_off = tl.arange(0, BLOCK)
        acc = 0.0
        for start_j in range(0, n, BLOCK):
            j = start_j + j_off
            mask = j > i
            uvals = tl.load(A_ptr + i * n + j, mask=mask)
            xvals = tl.load(b_ptr + j, mask=mask)
            acc += tl.sum(uvals * xvals, where=mask)
        rhs = (rhs - acc) / tl.load(A_ptr + i * n + i)
        tl.store(b_ptr + i, rhs)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    A_temp = A.clone().float().contiguous()
    b_temp = b.clone().float().contiguous()

    BLOCK_M = 64
    BLOCK_N = 64

    for k in range(n):
        # 1) Factor step for column k
        grid_factor = (triton.cdiv(n - (k+1), BLOCK_M),)
        _lu_factor_kernel[grid_factor](
            A_temp, n, k,
            BLOCK=BLOCK_M
        )

        # 2) Update step for columns > k
        for col_start in range(k+1, n, BLOCK_N):
            grid_update = (
                triton.cdiv(n - (k+1), BLOCK_M),
                triton.cdiv(n - col_start, BLOCK_N),
            )
            _lu_update_kernel[grid_update](
                A_temp, n, k, col_start,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N
            )

    # Forward substitution
    _forward_sub_kernel[(n,)](
        A_temp, b_temp, n,
        BLOCK_M
    )

    # Backward substitution
    _back_sub_kernel[(n,)](
        A_temp, b_temp, n,
        BLOCK_M
    )

    return b_temp.to(A.dtype)
