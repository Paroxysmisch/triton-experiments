import triton
import triton.language as tl


@triton.jit
def _kernel_apply_pivot_swap(
    ptr_matrix, ptr_piv, stride_row, stride_col, stride_piv,
    n, BLOCK: tl.constexpr
):
    """
    Kernel to swap rows based on pivot information.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK
    cols = tl.arange(0, BLOCK)
    for offset in range(BLOCK):
        row_idx = row_start + offset
        if row_idx < n:
            # Read pivot index
            piv_val = tl.load(ptr_piv + row_idx * stride_piv)
            pivot_idx = tl.to_int32(piv_val)

            # Swap rows row_idx and pivot_idx
            # Each row has n elements. We'll just swap within the current block range
            col = cols + 0
            mask = col < n
            val1 = tl.load(ptr_matrix + row_idx * stride_row + col * stride_col, mask=mask)
            val2 = tl.load(ptr_matrix + pivot_idx * stride_row + col * stride_col, mask=mask)
            tl.store(ptr_matrix + row_idx * stride_row + col * stride_col, val2, mask=mask)
            tl.store(ptr_matrix + pivot_idx * stride_row + col * stride_col, val1, mask=mask)


@triton.jit
def _kernel_lu_factor(
    ptr_matrix, ptr_piv, stride_row, stride_col, stride_piv,
    n, pivoting: tl.constexpr, BLOCK: tl.constexpr
):
    """
    Kernel to perform one step of the LU factorization with optional partial pivoting.
    This kernel is called repeatedly to factor all columns.
    """
    pid = tl.program_id(0)
    # Factor only the diagonal block for demonstration
    col = pid
    # If col >= n, we do nothing
    if col >= n:
        return

    # If pivoting, find pivot row
    if pivoting:
        # We do a simple pivot search in the diagonal column
        # (This is a naive approach for demonstration)
        best_idx = col
        max_val = tl.abs(tl.load(ptr_matrix + col * stride_row + col * stride_col))
        for r in range(col + 1, n):
            val = tl.abs(tl.load(ptr_matrix + r * stride_row + col * stride_col))
            cond = val > max_val
            max_val = tl.where(cond, val, max_val)
            best_idx = tl.where(cond, r, best_idx)
        # Write pivot index
        tl.store(ptr_piv + col * stride_piv, best_idx)

    # Factor out the pivot in the diagonal
    diag_val = tl.load(ptr_matrix + col * stride_row + col * stride_col)
    for r in range(col + 1, n):
        # Elimination step
        row_factor = tl.load(ptr_matrix + r * stride_row + col * stride_col) / diag_val
        tl.store(ptr_matrix + r * stride_row + col * stride_col, row_factor)
        # Eliminate subsequent columns
        for c in range(col + 1, n):
            val_rc = tl.load(ptr_matrix + r * stride_row + c * stride_col)
            val_uc = tl.load(ptr_matrix + col * stride_row + c * stride_col)
            new_val = val_rc - row_factor * val_uc
            tl.store(ptr_matrix + r * stride_row + c * stride_col, new_val)


@triton.jit
def _kernel_forward_solve(
    ptr_matrix, ptr_identity, stride_m_row, stride_m_col,
    stride_i_row, stride_i_col,
    n, BLOCK: tl.constexpr
):
    """
    Kernel to solve L y = P (or L y = I for sequential approach).
    This is a naive forward substitution on a single block diagonal.
    """
    bid = tl.program_id(0)
    row_start = bid * BLOCK
    rows = tl.arange(0, BLOCK)
    row = row_start + rows
    for offset in range(BLOCK):
        r = row_start + offset
        if r < n:
            # Solve L portion
            sum_val = tl.load(ptr_identity + r * stride_i_row, mask=True)
            for c in range(r):
                lrc = tl.load(ptr_matrix + r * stride_m_row + c * stride_m_col)
                i_c = tl.load(ptr_identity + c * stride_i_row, mask=True)
                sum_val = sum_val - lrc * i_c
            # L is unit-lower-triangular in standard LU, so no divide needed if diag is 1 in L.
            tl.store(ptr_identity + r * stride_i_row, sum_val, mask=True)


@triton.jit
def _kernel_backward_solve(
    ptr_matrix, ptr_identity, stride_m_row, stride_m_col,
    stride_i_row, stride_i_col,
    n, BLOCK: tl.constexpr
):
    """
    Kernel to solve U x = y (or U x = result_of_forward_solve).
    Naive backward substitution on a single block diagonal.
    """
    bid = tl.program_id(0)
    row_start = bid * BLOCK
    rows = tl.arange(0, BLOCK)
    row = row_start + rows
    for offset in reversed(range(BLOCK)):
        r = row_start + offset
        if r < n:
            sum_val = tl.load(ptr_identity + r * stride_i_row, mask=True)
            diag = tl.load(ptr_matrix + r * stride_m_row + r * stride_m_col)
            for c in range(r + 1, n):
                urc = tl.load(ptr_matrix + r * stride_m_row + c * stride_m_col)
                i_c = tl.load(ptr_identity + c * stride_i_row, mask=True)
                sum_val -= urc * i_c
            x_val = sum_val / diag
            tl.store(ptr_identity + r * stride_i_row, x_val, mask=True)


def invert_matrix_lu(A, *, pivot=True, out=None):
    """
    Invert a square matrix (or batch of matrices) using LU decomposition.
    A: Input tensor (float, double, cfloat, cdouble) of shape (..., n, n)
    pivot: bool, default=True, whether to use partial pivoting
    out: optional output tensor, ignored if None

    Returns: Tensor containing the inverse of A
    """
    import torch

    # Check that A is at least 2D and square
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("A must be a square matrix or a batch of square matrices.")

    n = A.shape[-1]
    batch_shape = A.shape[:-2]

    # If out is None, create a new tensor to hold the inverse
    if out is None:
        inv_shape = A.shape
        out = torch.empty_like(A)

    # Move data to GPU if not already
    A_working = A.clone().detach()  # to avoid modifying original
    device = A_working.device
    A_working = A_working.contiguous()
    out = out.contiguous()

    # We store pivot indices in an auxiliary buffer if pivot=True
    pivot_buf = None
    if pivot:
        pivot_buf = torch.empty((*batch_shape, n), dtype=torch.int32, device=device)

    # Flatten batch dims for processing in a loop
    flat_size = 1
    for b in batch_shape:
        flat_size *= b
    A_view = A_working.view(flat_size, n, n)
    if pivot:
        pivot_view = pivot_buf.view(flat_size, n) if pivot_buf is not None else None
    out_view = out.view(flat_size, n, n)

    # BLOCK size for kernels (naive approach)
    BLOCK = 1

    # LU Factorization with optional pivoting
    for idx in range(flat_size):
        # If pivoting, we must do partial pivot and row swaps
        for col in range(n):
            # 1) partial pivot step
            if pivot:
                _kernel_lu_factor[
                    1
                ](
                    A_view[idx],
                    pivot_view[idx],
                    A_view.stride(0),
                    A_view.stride(1),
                    pivot_view.stride(0),
                    n,
                    True,
                    BLOCK=BLOCK
                )
                # 2) swap row according to pivot
                _kernel_apply_pivot_swap[
                    1
                ](
                    A_view[idx],
                    pivot_view[idx],
                    A_view.stride(0),
                    A_view.stride(1),
                    pivot_view.stride(0),
                    n,
                    BLOCK=BLOCK
                )
            else:
                _kernel_lu_factor[
                    1
                ](
                    A_view[idx],
                    None,
                    A_view.stride(0),
                    A_view.stride(1),
                    0,
                    n,
                    False,
                    BLOCK=BLOCK
                )

        # Now A_view[idx] is factored into L and U in-place (with pivot info if pivot=True).
        # Next step is to solve for the inverse. We'll do that by columns of the identity.
        for col in range(n):
            # Prepare identity column
            e = torch.zeros(n, dtype=A.dtype, device=device)
            e[col] = 1
            # We do forward solve with L, then backward solve with U
            # 1) forward solve
            temp = e.clone()
            _kernel_forward_solve[
                1
            ](
                A_view[idx],
                temp,
                A_view.stride(0),
                A_view.stride(1),
                temp.stride(0),
                1,
                n,
                BLOCK=BLOCK
            )
            # 2) backward solve
            _kernel_backward_solve[
                1
            ](
                A_view[idx],
                temp,
                A_view.stride(0),
                A_view.stride(1),
                temp.stride(0),
                1,
                n,
                BLOCK=BLOCK
            )
            # Store result as column in out_view
            out_view[idx, :, col] = temp

    # Reshape out to the original shape
    out = out_view.view(*batch_shape, n, n)
    return out
