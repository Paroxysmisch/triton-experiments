import torch
import triton
import triton.language as tl


@triton.jit
def _conj_transpose_matmul_kernel(
    Q_ptr, B_ptr, C_ptr,
    M, N, K,  # Dimensions: Q is (M, N), B is (M, K), C is (N, K) for Q^H B
    stride_qm, stride_qn,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Computes C = Q^H * B in blocks:
      - Q has shape (M, N)
      - B has shape (M, K)
      - C has shape (N, K)
      Q^H is (N, M), so the multiplication is (N x M) * (M x K) -> (N x K).
    """
    # Program IDs for the 2D launch grid
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    # Block ranges
    row_range = row_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_range = col_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create an accumulator
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks of size BLOCK_SIZE_K
    for k_block in range(0, M, BLOCK_SIZE_K):
        k_offsets = k_block + tl.arange(0, BLOCK_SIZE_K)
        # Load Q block: indexing Q in conj-transpose => Q[row, col] -> Q[col, row]
        # but we keep real multiplication, so we handle it as Q_ptr + offset in memory
        # with row_range as 'n' dimension and k_offsets as 'm' dimension from Q^H perspective
        q_ptrs = Q_ptr + (k_offsets[:, None] * stride_qm) + (row_range[None, :] * stride_qn)
        q_block = tl.load(q_ptrs, mask=(k_offsets[:, None] < M) & (row_range[None, :] < N), other=0.0)

        # Load B block
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bm) + (col_range[None, :] * stride_bk)
        b_block = tl.load(b_ptrs, mask=(k_offsets[:, None] < M) & (col_range[None, :] < K), other=0.0)

        # Accumulate
        accum += tl.dot(q_block.to(tl.float32).T, b_block.to(tl.float32))

    # Write output
    c_ptrs = C_ptr + (row_range[:, None] * stride_cm) + (col_range[None, :] * stride_ck)
    mask = (row_range[:, None] < N) & (col_range[None, :] < K)
    tl.store(c_ptrs, accum, mask=mask)


@triton.jit
def _upper_tri_solve_kernel(
    R_ptr, C_ptr, X_ptr,
    N, K,  # R is (N, N), C is (N, K), solve R * X = C
    stride_rm, stride_rn,
    stride_cm, stride_ck,
    stride_xm, stride_xk
):
    """
    A naive kernel to solve an upper-triangular system R * X = C for each column in C.
    This kernel processes one row at a time in a single program_id to keep it simple.
    Not optimized for large N; for demo purposes only.
    """
    col_id = tl.program_id(0)
    # Each program handles a single column in [0..K)
    if col_id >= K:
        return

    # Offsets for the column col_id
    R_col_ptr = R_ptr
    C_col_ptr = C_ptr + col_id * stride_ck
    X_col_ptr = X_ptr + col_id * stride_xk

    # Solve in-place X = R^-1 C
    # BFS approach: x[n-1] = c[n-1]/r[n-1, n-1], then upwards
    for i in range(N - 1, -1, -1):
        # index in memory
        r_diag_off = i * stride_rm + i * stride_rn
        r_diag = tl.load(R_col_ptr + r_diag_off)
        c_val_off = i * stride_cm
        c_val = tl.load(C_col_ptr + c_val_off)

        # sum_{j=i+1..N-1} R[i, j] * X[j]
        sum_val = 0.0
        for j in range(i + 1, N):
            r_off = i * stride_rm + j * stride_rn
            x_off = j * stride_xm
            r_val = tl.load(R_col_ptr + r_off)
            x_val = tl.load(X_col_ptr + x_off)
            sum_val += r_val * x_val

        x_i = (c_val - sum_val) / r_diag
        # store X[i]
        x_off = i * stride_xm
        tl.store(X_col_ptr + x_off, x_i)


def least_squares_qr(A, b, *, mode='reduced', out=None):
    """
    Solves the least squares problem for an
