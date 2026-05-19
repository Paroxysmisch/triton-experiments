import torch
import triton
import triton.language as tl

@triton.jit
def _householder_norm_kernel(
    col_ptr, stride, length, BLOCK_SIZE: tl.constexpr
):
    """
    Compute the Euclidean norm of a column segment in parallel
    for building the Householder reflector.
    """
    pid = tl.program_id(0)
    # Each program handles one chunk of the column
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length

    # Load data
    data = tl.load(col_ptr + offsets * stride, mask=mask, other=0.0)
    # Square and sum partial results
    partial_sum = tl.sum(data * data, axis=0)

    # Use atomic_add to accumulate partial sums
    # in the first element of the column
    tl.atomic_add(col_ptr, partial_sum)


@triton.jit
def _householder_apply_kernel(
    mat_ptr, stride_row, stride_col,
    length_row, length_col,
    v_ptr, v_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Apply a Householder transformation: A = A - 2 * v * (v^T A)
    where v is the Householder vector. This kernel multiplies
    a sub-block of A by the Householder vector.
    """
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_offsets = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create 2D indices
    # We'll gather a sub-block: (row_offsets, col_offsets) from A
    # and apply the reflection
    rm = row_offsets[:, None]
    cn = col_offsets[None, :]

    # Mask for in-bounds
    rm_mask = rm < length_row
    cn_mask = cn < length_col
    mask2d = rm_mask & cn_mask

    # Load sub-block of A
    # mat_ptr + (rm * stride_row + cn * stride_col)
    ptrs = mat_ptr + rm * stride_row + cn * stride_col
    a_sub = tl.load(ptrs, mask=mask2d, other=0.0)

    # v^T * a_sub (applied to columns) => sum over row dimension
    # Expand v to the sub-block shape
    v_vals = tl.load(v_ptr + rm * v_stride, mask=rm_mask, other=0.0)
    dot_res = tl.sum(v_vals * a_sub, axis=0)

    # 2 * v * dot_res for each row
    # v_vals is shape (BLOCK_M, 1), dot_res is shape (1, BLOCK_N)
    # broadcast to shape (BLOCK_M, BLOCK_N)
    two = tl.full([1], 2.0, dtype=a_sub.dtype)
    reflect = two * v_vals * dot_res

    # A - reflect
    out_val = a_sub - reflect
    tl.store(ptrs, out_val, mask=mask2d)


def qr(A, mode='reduced', *, out=None):
    """
    qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor)

    Computes the QR decomposition of a matrix (or batch of matrices) A.
    mode: one of 'reduced', 'complete', or 'r'.
          'reduced' returns (Q, R) with Q of shape (..., m, k), R of shape (..., k, n)
          'complete' returns (Q, R) with Q of shape (..., m, m), R of shape (..., m, n)
          'r' returns (empty, R) with R of shape (..., k, n)
    out: optional output tuple of two tensors. Ignored if None.
    """
    # We'll do a simple batch-friendly Householder-based approach in Python,
    # dispatching some parallel subroutines to Triton kernels where possible.

    # Check input
    if mode not in ('reduced', 'complete', 'r'):
        raise ValueError("mode must be one of 'reduced', 'complete', or 'r'.")

    # Constructing shapes and handle batch dims
    original_shape = A.shape
    # A is of shape (..., m, n)
    batch_dims = original_shape[:-2]
    m = A.size(-2)
    n = A.size(-1)
    batch_size = 1
    for b in batch_dims:
        batch_size *= b

    # Flatten the batch dimensions for iteration
    A_2d = A.reshape(batch_size, m, n).clone()

    # Prepare Q and R placeholders
    # For 'r' mode we won't keep any Q, but let's set up for potential usage
    if mode == 'r':
        # Q is empty
        Q_shape = (batch_size, 0, 0)
    elif mode == 'reduced':
        # Q is shape (batch_size, m, min(m, n))
        Q_shape = (batch_size, m, min(m, n))
    else:  # 'complete'
        # Q is shape (batch_size, m, m)
        Q_shape = (batch_size, m, m)
    Q = A.new_empty(Q_shape)

    R = A_2d.clone()

    # Set Q to identity if not 'r'
    if mode != 'r':
        if mode == 'reduced':
            # Fill smaller Q with identity in the left portion
            k = min(m, n)
            Q.zero_()
            for b in range(batch_size):
                for i in range(k):
                    Q[b, i, i] = 1.0
        else:  # 'complete'
            Q.zero_()
            for b in range(batch_size):
                for i in range(m):
                    Q[b, i, i] = 1.0

    # Householder factorization (naive CPU loops, partial Triton usage)
    k = min(m, n) if mode != 'r' else min(m, n)
    BLOCK_SIZE = 128  # for column norm computations
    for b in range(batch_size):
        # We do a naive Householder transformation approach
        for i in range(k):
            # 1) Compute norm of the sub-column R[b, i:, i]
            sub_len = m - i
            col_ptr = R[b, i:, i].data_ptr()  # offset pointer
            # Zero out the first element of col to store the partial sums
            R[b, i, i] = 0.0

            # launch kernel to accumulate sum of squares in R[b, i, i]
            grid = ( (sub_len + BLOCK_SIZE - 1) // BLOCK_SIZE, )
            _householder_norm_kernel[grid](
                col_ptr, R.stride(-2), sub_len,
                BLOCK_SIZE=BLOCK_SIZE
            )
            # now R[b, i, i] holds the partial sum of squares
            norm_val = (R[b, i, i].item())**0.5

            # 2) Build reflection vector v
            alpha = -torch.sign(R[b, i, i]) * norm_val if R[b, i, i] != 0 else -norm_val
            v = R[b, i:, i].clone()
            v[0] = v[0] - alpha
            denom = torch.norm(v)
            if denom != 0:
                v = v / denom

            # 3) Apply Householder to R (i-th column onward, i-th row onward)
            # parallel block updates
            if denom != 0:
                grid_m = ( (m - i + 31) // 32 )
                grid_n = ( (n - i + 31) // 32 )
                grid2d = (grid_m, grid_n)
                _householder_apply_kernel[grid2d](
                    R[b, i:, i:].data_ptr(),
                    R.stride(-2), R.stride(-1),
                    m - i, n - i,
                    v.data_ptr(), 1,
                    BLOCK_M=32, BLOCK_N=32
                )

            # 4) Apply Householder to Q if not 'r'
            if mode != 'r' and denom != 0:
                # dimension of Q is either (m, k) or (m, m)
                # If 'reduced', Q is (m, k) => we apply reflection to columns 0..k-1
                # If 'complete', Q is (m, m) => all columns
                apply_cols = (Q.size(-1) if mode == 'complete' else min(m, n))
                grid_m = ( (m + 31) // 32 )
                grid_n = ( (apply_cols + 31) // 32 )
                grid2d = (grid_m, grid_n)
                _householder_apply_kernel[grid2d](
                    Q[b].data_ptr(),
                    Q.stride(-2), Q.stride(-1),
                    m, apply_cols,
                    v.data_ptr(), 1,
                    BLOCK_M=32, BLOCK_N=32
                )

            # restore R[b, i, i] to alpha
            R[b, i, i] = alpha

    # Now R is upper-triangular, Q is built as product of reflectors
    # Reshape back to original batch shape
    if mode == 'r':
        # Q is empty
        Q_final = A.new_empty((*batch_dims, 0,
