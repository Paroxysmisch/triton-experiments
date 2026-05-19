import torch
import triton
import triton.language as tl
import torch.linalg as linalg

@triton.jit
def lu_solve_kernel(
    L, U, P, b, x, n, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)

    # Load rows of L, U, P, and b
    L_row = tl.load(L + row[:, None] * n + col[None, :], mask=row[:, None] < n)
    U_row = tl.load(U + row[:, None] * n + col[None, :], mask=row[:, None] < n)
    P_row = tl.load(P + row[:, None] * n + col[None, :], mask=row[:, None] < n)
    b_row = tl.load(b + row, mask=row < n)

    # Apply permutation matrix P to b
    Pb = tl.dot(P_row, b_row)

    # Forward substitution to solve L * y = Pb
    y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        y[i] = (Pb[i] - tl.dot(L_row[i, :i], y[:i])) / L_row[i, i]

    # Backward substitution to solve U * x = y
    for i in range(BLOCK_SIZE - 1, -1, -1):
        x[i] = (y[i] - tl.dot(U_row[i, i + 1:], x[i + 1:])) / U_row[i, i]

    # Store result in x
    tl.store(x + row, x, mask=row < n)

def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix A must be square"
    assert b.shape == (n,), "Vector b must have shape (n,)"

    # Perform LU decomposition
    P, L, U = linalg.lu(A)

    # Prepare output tensor
    x = torch.empty_like(b)

    # Define grid and block sizes
    BLOCK_SIZE = 128
    grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)

    # Launch the kernel
    lu_solve_kernel[grid](
        L, U, P, b, x, n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return x
