import triton
from triton.language import *

@triton.jit
def cholesky_decomposition_kernel(
    A_ptr, L_ptr, n,
    BLOCK_SIZE: int = 32
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n, BLOCK_SIZE)

    # Each thread computes one element of L
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = row

    A = tl.load(A_ptr + row[:, None] * n + col)
    L = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Compute L[row, col]
    for i in range(col + 1):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * L[j, col]
        L[i, col] = (A[i, col] - sum_val) / L[i, i]

    tl.store(L_ptr + row[:, None] * n + col, L)

@triton.jit
def forward_kernel(
    L_ptr, b_ptr, x_ptr, n, k,
    BLOCK_SIZE: int = 32
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(k, BLOCK_SIZE)

    # Each thread computes one element of x
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = row

    L = tl.load(L_ptr + row[:, None] * n + col)
    b = tl.load(b_ptr + row[:, None] * n + col)
    x = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)

    # Forward substitution
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += L[i, j] * x[j, :]
        x[i, :] = (b[i, :] - sum_val) / L[i, i]

    tl.store(x_ptr + row[:, None] * n + col, x)

@triton.jit
def backward_kernel(
    L_ptr, x_ptr, y_ptr, n, k,
    BLOCK_SIZE: int = 32
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(k, BLOCK_SIZE)

    # Each thread computes one element of y
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = row

    L = tl.load(L_ptr + row[:, None] * n + col)
    x = tl.load(x_ptr + row[:, None] * n + col)
    y = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)

    # Backward substitution
    for i in reversed(range(n)):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += L[j, i] * y[j, :]
        y[i, :] = (x[i, :] - sum_val) / L[i, i]

    tl.store(y_ptr + row[:, None] * n + col, y)

@triton.jit
def fused_cholesky_solve(A_ptr, b_ptr, x_ptr, n, k,
                          BLOCK_SIZE: int = 32):
    # Step 1: Cholesky Decomposition
    cholesky_decomposition_kernel[A, B](A_ptr, x_ptr, n, BLOCK_SIZE)

    # Step 2: Solve Ly = b
    forward_kernel[B, C](x_ptr, b_ptr, x_ptr, n, k, BLOCK_SIZE)

    # Step 3: Solve L^T x = y
    backward_kernel[B, D](x_ptr, x_ptr, x_ptr, n, k, BLOCK_SIZE)
