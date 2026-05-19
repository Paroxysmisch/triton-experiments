import triton
import triton.language as tl
import torch

@triton.jit
def _lu_decomposition_kernel(
    A_ptr,       # (n, n) matrix to factor (in/out)
    P_ptr,       # (n,) pivot vector (out)
    N,           # dimension n
    pivot_flag: tl.constexpr
):
    """
    In-place LU decomposition with optional partial pivoting for a single matrix.
    A_ptr points to the matrix A on entry, and on exit it stores both L and U factors:
      - L is stored in the strict lower-triangular part (without unit diagonals)
      - U is stored in the upper-triangular part (including diagonal)
    If pivot_flag is True, partial pivot indices are written to P_ptr.
    """
    # We launch this kernel with a single program_id(0) for simplicity (one block).
    pid = tl.program_id(0)
    # Only proceed if pid == 0
    if pid != 0:
        return

    # We'll do a naive O(n^3) in-place factorization. For demonstration only.
    # Offsets in global memory:
    # A(row, col) = A_ptr + row*N + col
    for i in range(0, N):
        # If pivoting is enabled, find pivot = argmax |A[j, i]| for j >= i
        if pivot_flag:
            # Find pivot row
            max_idx = i
            max_val = tl.abs(tl.load(A_ptr + i*N + i))
            for j in range(i+1, N):
                cand = tl.abs(tl.load(A_ptr + j*N + i))
                max_idx = tl.where(cand > max_val, j, max_idx)
                max_val = tl.where(cand > max_val, cand, max_val)
            # Write pivot index for row i
            tl.store(P_ptr + i, max_idx)

            # If pivot row != i, swap them
            do_swap = max_idx != i
            if do_swap:
                for c in range(0, N):
                    x_i = tl.load(A_ptr + i*N + c)
                    x_m = tl.load(A_ptr + max_idx*N + c)
                    tl.store(A_ptr + i*N + c, x_m)
                    tl.store(A_ptr + max_idx*N + c, x_i)
        else:
            # Write trivial pivot (no pivoting)
            tl.store(P_ptr + i, i)

        # Now pivot row is i; proceed with factorization
        # A[i, i] is pivot element
        pivot_val = tl.load(A_ptr + i*N + i)

        # If pivot_val == 0, factorization fails in practice, but we'll skip checks
        for r in range(i+1, N):
            elem = tl.load(A_ptr + r*N + i)
            # L factor = A[r, i] / pivot
            new_val = elem / pivot_val
            tl.store(A_ptr + r*N + i, new_val)  # store L(r, i)
            # Eliminate from row r
            for c in range(i+1, N):
                rc_val = tl.load(A_ptr + r*N + c)
                ic_val = tl.load(A_ptr + i*N + c)
                update = rc_val - new_val * ic_val
                tl.store(A_ptr + r*N + c, update)


@triton.jit
def _apply_permutation_kernel(
    P_ptr,        # (n,) pivot vector
    B_ptr,        # (n, k) original RHS
    Out_ptr,      # (n, k) permuted RHS (out)
    N,            # dimension n
    K             # number of RHS columns k
):
    """
    Apply the stored pivot data P to B, writing to Out.
    We do out[i, :] = B[p[i], :].
    Launch with 2D grid: (n, k). Each thread handles one (row, col).
    """
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    # Single element indices
    row_ix = row_pid
    col_ix = col_pid

    if row_ix < N and col_ix < K:
        pivot_val = tl.load(P_ptr + row_ix, mask=row_ix < N)
        # pivot_val is the row from which data must be taken
        b_elem = tl.load(B_ptr + pivot_val*K + col_ix, mask=(pivot_val < N) & (col_ix < K))
        # store result
        tl.store(Out_ptr + row_ix*K + col_ix, b_elem, mask=(row_ix < N) & (col_ix < K))


@triton.jit
def _forward_substitution_kernel(
    A_ptr,   # (n, n) combined L and U, L in lower part
    B_ptr,   # (n, k) in/out => overwritten with the solution y after L*y = B
    N,       # dimension n
    K        # number of RHS columns
):
    """
    Solve L*y = B for y, in-place in B.
    L is stored in A's lower triangle (unit diagonal implied).
    We do an elementary forward-substitution for each column of B.
    Launch with 2D grid: (k, 1) or (1, k), and each program_id does one column.
    """
    col_pid = tl.program_id(0)
    if col_pid >= K:
        return

    for i in range(0, N):
        # B[i, col] = B[i, col] - L(i,0..i-1) dot B(0..i-1, col)
        # L(i,i) = 1.0 implied
        sum_val = tl.load(B_ptr + i*K + col_pid)
        for j in range(0, i):
            L_ij = tl.load(A_ptr + i*N + j)  # L(i,j)
            b_j = tl.load(B_ptr + j*K + col_pid)
            sum_val -= L_ij * b_j
        tl.store(B_ptr + i*K + col_pid, sum_val)


@triton.jit
def _backward_substitution_kernel(
    A_ptr,   # (n, n) combined L and U, U in upper triangle
    B_ptr,   # (n, k) in/out => overwritten with final solution x after U*x = B
    N,       # dimension n
    K        # number of RHS columns
):
    """
    Solve U*x = B for x, in-place in B.
    U is stored in A's upper triangle (including diagonal).
    We do an elementary backward-substitution for each column of B.
    Launch with 2D grid: (k, 1) or (1, k), and each program_id does one column.
    """
    col_pid = tl.program_id(0)
    if col_pid >= K:
        return

    for i in range(N-1, -1, -1):
        # B[i, col] = (B[i, col] - sum_{j=i+1..n-1} U(i,j)*B[j, col]) / U(i,i)
        sum_val = tl.load(B_ptr + i*K + col_pid)
        pivot_val = tl.load(A_ptr + i*N + i)  # U(i,i)
        for j in range(i+1, N):
            U_ij = tl.load(A_ptr + i*N + j)
            b_j = tl.load(B_ptr + j*K + col_pid)
            sum_val -= U_ij * b_j
        x_val = sum_val / pivot_val
        tl.store(B_ptr + i*K + col_pid, x_val)


def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    """
    Solves A * X = Bs for X, where A is (*, n, n) and Bs is (*, n, k).
    Optionally uses partial pivoting if pivot=True.
    Returns a tensor of shape (*, n, k), or writes into 'out' if provided.
    """
    # Check shapes
    if A.ndim < 2:
        raise ValueError("A must have at least 2 dimensions")
    if Bs.ndim < 2:
        raise ValueError("Bs must have at least 2 dimensions")
    *batch_dimsA, nA, nA2 = A.shape
    *batch_dimsB, nB,
