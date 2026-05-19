import triton
import triton.language as tl


# ------------------------------------------------------------------------------
# KERNELS
# ------------------------------------------------------------------------------
@triton.jit
def _row_swap_kernel(
    ptr_matrix, stride_b, stride_m, stride_n,
    n, row1, row2,
    **meta
):
    """
    Swaps two rows 'row1' and 'row2' in a single matrix.
    Used for partial pivoting.
    """
    pid = tl.program_id(0)
    # Each program handles one row swap
    # Because we only swap a single pair of rows, we do it in one launch
    # across columns
    col = pid
    if col < n:
        # Each column in the given row
        off1 = row1 * stride_m + col * stride_n
        off2 = row2 * stride_m + col * stride_n
        val1 = tl.load(ptr_matrix + off1)
        val2 = tl.load(ptr_matrix + off2)
        tl.store(ptr_matrix + off1, val2)
        tl.store(ptr_matrix + off2, val1)


@triton.jit
def _elimination_step_kernel(
    ptr_matrix, stride_b, stride_m, stride_n,
    n, pivot_idx,
    **meta
):
    """
    Performs the elimination step for LU factorization at pivot 'pivot_idx'.
    Updates the rest of the matrix below pivot_idx.
    """
    # Each program handles one row
    row = tl.program_id(0)

    if row > pivot_idx and row < n:
        # Load pivot
        pivot_val = tl.load(ptr_matrix + pivot_idx * stride_m + pivot_idx * stride_n)
        # Load current row pivot element
        cur_val = tl.load(ptr_matrix + row * stride_m + pivot_idx * stride_n)
        # Compute multiplier
        factor = cur_val / pivot_val
        tl.store(ptr_matrix + row * stride_m + pivot_idx * stride_n, factor)

        # Eliminate columns to the right
        col = pivot_idx + 1
        while col < n:
            # A[row, col] -= factor * A[pivot_idx, col]
            a_rc = tl.load(ptr_matrix + row * stride_m + col * stride_n)
            a_pc = tl.load(ptr_matrix + pivot_idx * stride_m + col * stride_n)
            a_rc = a_rc - factor * a_pc
            tl.store(ptr_matrix + row * stride_m + col * stride_n, a_rc)
            col += 1


@triton.jit
def _forward_substitution_kernel(
    ptr_matrix, ptr_out, stride_mA, stride_nA, stride_mO, stride_nO,
    n,
    **meta
):
    """
    Solves L y = P (where P is embedded in the L matrix pivoted region),
    writing the partial result y = L^{-1} P into ptr_out.
    Assumes the lower part of ptr_matrix has the needed L factors (unit diagonal).
    This kernel works column by column for the identity basis to get the partial solution.
    """
    # Each program handles one row of the partial solve for the identity column
    row = tl.program_id(0)
    col = tl.program_id(1)
    if row < n and col < n:
        # Because we treat columns of identity as RHS
        # We proceed row by row in forward-substitution
        sum_val = 0.0
        # We'll do a naive loop
        for k in range(row):
            a_rk = tl.load(ptr_matrix + row * stride_mA + k * stride_nA)
            o_kc = tl.load(ptr_out + k * stride_mO + col * stride_nO)
            sum_val += a_rk * o_kc

        # The L diagonal is implicitly 1.0 in an LU factorization
        # so no division for the diagonal
        # Right-hand side is identity => delta_{row, col}
        rhs = 1.0 if (row == col) else 0.0

        val = rhs - sum_val
        tl.store(ptr_out + row * stride_mO + col * stride_nO, val)


@triton.jit
def _backward_substitution_kernel(
    ptr_matrix, ptr_inout, stride_mA, stride_nA, stride_mO, stride_nO,
    n,
    **meta
):
    """
    Solves U x = y (where y is in ptr_inout),
    writing the final inverse columns, i.e. x = U^{-1} y, back into ptr_inout.
    This kernel works column by column on the partial solved data in ptr_inout.
    """
    # Each program handles one row of the backward substitution for a column
    row = tl.program_id(0)
    col = tl.program_id(1)
    # We'll proceed in reverse order from bottom to top
    idx = n - 1 - row  # so row=0 => idx=n-1
    if idx >= 0 and idx < n and col < n:
        sum_val = 0.0
        # Accumulate partial sums from known upper-triangular region
        for k in range(idx + 1, n):
            a_ik = tl.load(ptr_matrix + idx * stride_mA + k * stride_nA)
            o_kc = tl.load(ptr_inout + k * stride_mO + col * stride_nO)
            sum_val += a_ik * o_kc

        diag = tl.load(ptr_matrix + idx * stride_mA + idx * stride_nA)
        rhs = tl.load(ptr_inout + idx * stride_mO + col * stride_nO)

        x_val = (rhs - sum_val) / diag
        tl.store(ptr_inout + idx * stride_mO + col * stride_nO, x_val)


# ------------------------------------------------------------------------------
# WRAPPER
# ------------------------------------------------------------------------------
def invert_matrix_lu(A, *, pivot=True, out=None):
    """
    invert_matrix_lu(A, *, pivot=True, out=None) -> Tensor

    Computes the inverse of a square matrix (or batch of square matrices)
    using LU decomposition. By default partial pivoting is used (pivot=True).
    If 'out' is not None, the result is stored there, otherwise
    a new tensor is returned.

    Math:
        A = P L U
        A^{-1} = U^{-1} L^{-1} P
        Y = L^{-1} P
        A^{-1} = U^{-1} Y
    """
    import torch

    # Handle batch dimension
    if A.dim() < 2:
        raise ValueError("Input must be at least 2D (matrix).")
    batch_dims = A.shape[:-2]
    n = A.shape[-2]
    if A.shape[-1] != n:
        raise ValueError("Input matrix must be square in the last two dimensions.")

    # Create output tensor if None
    if out is None:
        out = torch.empty_like(A)

    # Flatten batch for iteration
    num_mats = 1
    for s in batch_dims:
        num_mats *= s

    # Reshape A and out as [num_mats, n, n]
    A_reshaped = A.view(num_mats, n, n)
    out_reshaped = out.view(num_mats, n, n)

    # We'll do a naive loop over each matrix in the batch.
    for bidx in range(num_mats):
        # Copy input matrix to out_reshaped so we can factor in-place
        # We'll store final inverse also in out_reshaped
        out_reshaped[bidx].copy_(A_reshaped[bidx])

        # Perform naive LU factorization with partial pivoting if pivot=True
        for k in range(n):
            if pivot:
                # Find pivot row by maximum in column k
                pivot_vals = torch.abs(out_reshaped[bidx, k:, k])
                max_idx = torch.argmax(pivot_vals).item() + k
                if max_idx != k:
                    # Swap row k with row max_idx
                    triton.run(
                        _row_swap_kernel,
                        ranges=(n,),
                        num_warps=1,
                        args=[
                            out_reshaped[bidx],
                            n*n, n, 1,  # stride_b, stride_m, stride_n
                            n,
                            k,
                            max_idx
                        ]
                    )

            # Elimination step
            triton.run(
                _elimination_step_kernel,
                ranges=(n,),
                num_warps=1,
                args=[
                    out_reshaped[bidx],
                    n*n, n, 1,
                    n,
                    k
                ]
            )

        # Now out_reshaped[bidx] contains L and U packed in place.

        # We next solve for each column of the inverse in two steps:
        # forward-substitution and backward-substitution,
        # storing the final result in out_reshaped[bidx].
        # We'll do it in-place in another buffer so we don't overwrite partial results.
        eye_like = torch.zeros_like(out_reshaped[bidx])
        # We'll store partial solution in 'eye_like'
        # Then store final solution in out_reshaped[bidx]
        triton.run(
            _forward_substitution_kernel,
            ranges=(n, n),
            num_warps=1,
            args=[
                out_reshaped[bidx],
                eye_like,
                n, 1, n, 1,
                n
            ]
        )
        # Now 'eye_like' holds Y = L^{-1} * Identity columns

        # Next do backward substitution on eye_like
        triton.run(
            _backward_substitution_kernel,
            ranges=(n, n),
            num_warps=1,
            args=[
                out_reshaped[bidx],
                eye_like,
                n, 1, n, 1,
                n
