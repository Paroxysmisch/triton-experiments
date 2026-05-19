import torch
import triton
import triton.language as tl

# Kernel 1: In-place symmetric matrix multiplication update of C
#           C = alpha * (A @ A^T) + beta * C
@triton.jit
def _sym_mm_kernel(
    A_ptr,       # [n, m] input
    C_ptr,       # [n, n] output
    alpha,       # scalar
    beta,        # scalar
    n,           # n dimension
    m,           # m dimension
    strideA0,    # row stride of A
    strideA1,    # col stride of A
    strideC0,    # row stride of C
    strideC1,    # col stride of C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs for the 2D grid.
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    # Each block covers a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of C
    row_start = row_block * BLOCK_SIZE_M
    col_start = col_block * BLOCK_SIZE_N

    # For each thread in this block, build row/col offsets within the tile
    r_idxs = row_start + tl.arange(0, BLOCK_SIZE_M)
    c_idxs = col_start + tl.arange(0, BLOCK_SIZE_N)

    # Create a 2D mask to guard loads/stores
    rm = r_idxs < n
    cm = c_idxs < n

    # Initialize an accumulator for each element of the tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension in blocks of BLOCK_SIZE_K
    for k_block_start in range(0, m, BLOCK_SIZE_K):
        # Build the range of k for this chunk
        k_range = tl.arange(0, BLOCK_SIZE_K)
        k = k_block_start + k_range
        # Mask to handle remainder if m is not multiple of BLOCK_SIZE_K
        km = k < m

        # Load a block of A[row, k]
        a = tl.where(
            (rm[:, None] & km[None, :]),
            tl.load(A_ptr + r_idxs[:, None] * strideA0 + k[None, :] * strideA1),
            0.0
        )

        # Load a block of A[col, k] => effectively from A^T
        # So for "row" in A^T we want the old "k", and for "col" in A^T we want the old "col"
        # But the symmetrical multiplication means A[col, k] is A_ptr[col, k]
        b = tl.where(
            (cm[:, None] & km[None, :]),
            tl.load(A_ptr + c_idxs[:, None] * strideA0 + k[None, :] * strideA1),
            0.0
        )
        # b has shape (BLOCK_SIZE_N x BLOCK_SIZE_K), a has shape (BLOCK_SIZE_M x BLOCK_SIZE_K)
        # We want to multiply a (MxK) by b (NxK) over dimension K => accumulate in (MxN)
        # We'll transpose b in the dot: a @ b^T => but we can do elementwise multiply across K then sum
        # This is done by broadcasting a's shape [M, K] against b's shape [N, K].
        # We'll do a manual outer product over K:
        # acc[m_idx, n_idx] += sum_{k_idx}( a[m_idx, k_idx]*b[n_idx, k_idx] )
        # We can do:

        for kk in range(0, BLOCK_SIZE_K):
            # each iteration is one column from a, one column from b
            a_vec = a[:, kk]       # shape [BLOCK_SIZE_M]
            b_vec = b[:, kk]       # shape [BLOCK_SIZE_N]
            acc += a_vec[:, None] * b_vec[None, :]

    # Now scale the computed block of mm by alpha
    acc = acc * alpha

    # Load and add beta*C
    c_old = tl.where(
        (rm[:, None] & cm[None, :]),
        tl.load(C_ptr + r_idxs[:, None] * strideC0 + c_idxs[None, :] * strideC1),
        0.0
    )
    c_new = acc + beta * c_old

    # Store final block in C
    tl.store(
        C_ptr + r_idxs[:, None] * strideC0 + c_idxs[None, :] * strideC1,
        c_new,
        mask=(rm[:, None] & cm[None, :])
    )


# Kernel 2: partial sum of |C| into an output buffer per block
@triton.jit
def _sum_abs_kernel(
    C_ptr,        # [n, n] data
    out_ptr,      # [grid_size] partial sums
    n,            # dimension n
    strideC0,     # row stride for C
    strideC1,     # col stride for C
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n * n)

    # Flatten C (row
