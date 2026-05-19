import triton
import triton.language as tl

@triton.jit
def qr_decomposition_kernel(A_ptr, A_batch, A_m, A_n, Q_ptr, R_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch = pid // (A_m * A_n)
    row = (pid % (A_m * A_n)) // A_n
    col = (pid % (A_m * A_n)) % A_n

    A = tl.load(A_ptr + batch * A_m * A_n + row * A_n + col)
    Q = tl.zeros((A_m, A_m), dtype=tl.float32)
    R = tl.zeros((A_m, A_n), dtype=tl.float32)

    for i in range(A_n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = tl.sum(Q[:, j] * A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = tl.sqrt(tl.sum(v * v))
        Q[:, i] = v / R[i, i]

    tl.store(Q_ptr + batch * A_m * A_m + row * A_m + col, Q[row, col])
    tl.store(R_ptr + batch * A_m * A_n + row * A_n + col, R[row, col])

@triton.jit
def solve_least_squares_kernel(Q_ptr, R_ptr, b_ptr, A_batch, A_m, A_n, b_k, x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch = pid // (A_m * b_k)
    row = (pid % (A_m * b_k)) // b_k
    col = (pid % (A_m * b_k)) % b_k

    Q = tl.load(Q_ptr + batch * A_m * A_m + row * A_m + col)
    R = tl.load(R_ptr + batch * A_m * A_n + row * A_n + col)
    b = tl.load(b_ptr + batch * A_m * b_k + row * b_k + col)

    y = tl.zeros((A_m, b_k), dtype=tl.float32)
    for i in range(A_m):
        y[i] = tl.sum(Q[:, i] * b)

    x = tl.zeros((A_n, b_k), dtype=tl.float32)
    for i in range(A_n - 1, -1, -1):
        x[i] = (y[i] - tl.sum(R[i, i+1:] * x[i+1:])) / R[i, i]

    tl.store(x_ptr + batch * A_n * b_k + row * b_k + col, x[row, col])

import torch
import triton
import triton.language as tl

def least_squares_qr(A, b, *, mode='reduced', out=None) -> torch.Tensor:
    A = A.contiguous()
    b = b.contiguous()

    A_batch, A_m, A_n = A.shape
    b_batch, b_m, b_k = b.shape if b.dim() == 3 else (b.shape[0], b.shape[1], 1)

    assert A_batch == b_batch, "Batch dimensions of A and b must match"
    assert A_m == b_m, "Number of rows in A and b must match"

    if mode == 'reduced':
        Q_shape = (A_batch, A_m, A_m)
        R_shape = (A_batch, A_m, A_n)
    elif mode == 'complete':
        Q_shape = (A_batch, A_m, A_m)
        R_shape = (A_batch, A_m, A_n)
    else:
        raise ValueError("mode must be 'reduced' or 'complete'")

    Q = torch.empty(Q_shape, dtype=A.dtype, device=A.device)
    R = torch.empty(R_shape, dtype=A.dtype, device=A.device)

    grid = (A_batch * A_m * A_n, )
    qr_decomposition_kernel[grid](A, A_batch, A_m, A_n, Q, R, BLOCK_SIZE=128)

    x = torch.empty((A_batch, A_n, b_k), dtype=A.dtype, device=A.device) if out is None else out

    grid = (A_batch * A_m * b_k, )
    solve_least_squares_kernel[grid](Q, R, b, A_batch, A_m, A_n, b_k, x, BLOCK_SIZE=128)

    return x
