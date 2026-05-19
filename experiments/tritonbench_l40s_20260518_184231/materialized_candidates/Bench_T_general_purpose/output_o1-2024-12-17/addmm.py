import triton
import triton.language as tl
import math

@triton.jit
def _addmm_kernel(
    INPUT_PTR, MAT1_PTR, MAT2_PTR, OUT_PTR,
    N, M, P,
    STRIDE_IN_ROWS, STRIDE_IN_COLS,
    STRIDE_M1_ROWS, STRIDE_M1_COLS,
    STRIDE_M2_ROWS, STRIDE_M2_COLS,
    STRIDE_OUT_ROWS, STRIDE_OUT_COLS,
    ALPHA, BETA,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    row_idx = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    # Create a pointer to out tile
    # We'll compute partial sums in fp32 to reduce numeric issues
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # If BETA != 0, read from input and scale by BETA
    if BETA != 0.0:
        in_rows = tl.where(row_idx < N, row_idx, 0)
        in_cols = tl.where(col_idx < P, col_idx, 0)
        in_offs = in_rows * STRIDE_IN_ROWS + in_cols * STRIDE_IN_COLS
        acc += BETA * tl.load(INPUT_PTR + in_offs, mask=(row_idx < N)[:, None] & (col_idx < P)[None, :], other=0.0)

    # Matrix multiply mat1 (N x M) by mat2 (M x P)
    # Split K dimension into blocks of BLOCK_K
    for k_block_start in range(0, M, BLOCK_K):
        k_range = tl.arange(0, BLOCK_K)
        k_idx = k_block_start + k_range

        # Rows from mat1
        m1_rows = tl.where(row_idx < N, row_idx, 0)
        m1_cols = tl.where(k_idx < M, k_idx, 0)
        # Cols from mat2
        m2_rows = m1_cols
        m2_cols = tl.where(col_idx < P, col_idx, 0)

        # Compute pointer offsets
        m1_offs = m1_rows[:, None] * STRIDE_M1_ROWS + m1_cols[None, :] * STRIDE_M1_COLS
        m2_offs = m2_rows[:, None] * STRIDE_M2_ROWS + m2_cols[None, :] * STRIDE_M2_COLS

        # Load
        a = tl.load(MAT1_PTR + m1_offs, mask=(row_idx < N)[:, None] & (k_idx < M)[None, :], other=0.0)
        b = tl.load(MAT2_PTR + m2_offs, mask=(k_idx < M)[:, None] & (col_idx < P)[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Multiply accumulated matrix by ALPHA
    acc = acc * ALPHA

    # Write out
    out_rows = tl.where(row_idx < N, row_idx, 0)
    out_cols = tl.where(col_idx < P, col_idx, 0)
    out_offs = out_rows * STRIDE_OUT_ROWS + out_cols * STRIDE_OUT_COLS
    tl.store(OUT_PTR + out_offs, acc, mask=(row_idx < N)[:, None] & (col_idx < P)[None, :])


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    """
    out = β * input + α * (mat1 @ mat2)
    """
    # Shapes
    n1, m1 = mat1.shape
    m2, p2 = mat2.shape
    if m1 != m2:
        raise ValueError("mat1 and mat2 shapes are not compatible for matrix multiplication.")

    N, M, P = n1, m1, p2
    # Handle broadcast or check if input can be broadcast to (N x P)
    # Simple check: if input.shape != (N, P), we either expand or verify it's broadcastable
    if input.shape != (N, P):
        # naive broadcast check
        if not all(
            (i == o or i == 1) for i, o in zip(input.shape[::-1], (P, N)[::-1])
        ):
            raise ValueError("input is not broadcastable to (N, P)")

    # Allocate out if none
    if out is None:
        out = input.new_empty((N, P))

    # Strides
    stride_in_rows, stride_in_cols = (0, 0)
    if input.dim() == 2:
        stride_in_rows = input.stride(0)
        stride_in_cols = input.stride(1)

    stride_m1_rows, stride_m1_cols = mat1.stride(0), mat1.stride(1)
    stride_m2_rows, stride_m2_cols = mat2.stride(0), mat2.stride(1)
    stride_out_rows, stride_out_cols = out.stride(0), out.stride(1)

    # Grid
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (
        math.ceil(N / BLOCK_M),
        math.ceil(P / BLOCK_N),
    )

    _addmm_kernel[grid](
        input, mat1, mat2, out,
        N, M, P,
        stride_in_rows, stride_in_cols,
        stride_m1_rows, stride_m1_cols,
        stride_m2_rows, stride_m2_cols,
        stride_out_rows, stride_out_cols,
        alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
