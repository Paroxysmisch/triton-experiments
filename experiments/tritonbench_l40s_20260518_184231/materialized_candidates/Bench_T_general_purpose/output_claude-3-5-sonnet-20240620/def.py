@triton.jit
def lu_decomposition(A, P, L, U, n):
    # LU decomposition with partial pivoting
    # ... existing code for LU decomposition ...
    pass

@triton.jit
def solve_lu(L, U, b, n, k):
    # Solve L y = b and U x = y
    # ... existing code for solving ...
    pass

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> Tensor:
    # Validate input shapes
    # ... existing code for validation ...

    # Get batch dimensions and sizes
    batch_size = A.shape[0] if A.ndim > 2 else 1
    n = A.shape[-1]
    k = Bs.shape[-1]

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(Bs)

    # Allocate memory for L, U, and P
    L = torch.empty((batch_size, n, n), dtype=A.dtype)
    U = torch.empty((batch_size, n, n), dtype=A.dtype)
    P = torch.empty((batch_size, n, n), dtype=A.dtype)

    # Perform LU decomposition
    lu_decomposition(A, P, L, U, n)

    # Solve for each right-hand side
    for i in range(k):
        b_i = Bs[..., i]
        y_i = solve_lu(L, U, b_i, n, k)
        out[..., i] = y_i

    return out
