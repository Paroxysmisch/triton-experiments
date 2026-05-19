import triton
import triton.language as tl

@triton.jit
def qr_decomposition_kernel(A_ptr, R_ptr, Q_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load the matrix A
    A = tl.load(A_ptr + block_start * n + tl.arange(0, n))

    # QR Decomposition
    for i in range(n):
        # Compute the norm of the i-th column
        norm = tl.sqrt(tl.sum(A[:, i] * A[:, i]))
        R = tl.zeros((n, n), dtype=tl.float32)
        R[i, i] = norm

        # Compute the i-th column of Q
        Q = A[:, i] / norm

        # Update A
        for j in range(i + 1, n):
            R[i, j] = tl.sum(Q * A[:, j])
            A[:, j] -= R[i, j] * Q

    # Store the results
    tl.store(R_ptr + block_start * n + tl.arange(0, n), R)
    tl.store(Q_ptr + block_start * n + tl.arange(0, n), Q)

@triton.jit
def solve_linear_system_kernel(Q_ptr, R_ptr, b_ptr, x_ptr, m, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load the matrix Q and R
    Q = tl.load(Q_ptr + block_start * n + tl.arange(0, n))
    R = tl.load(R_ptr + block_start * n + tl.arange(0, n))

    # Load the vector b
    b = tl.load(b_ptr + block_start * k + tl.arange(0, k))

    # Compute Q^T * b
    Q_T = tl.transpose(Q)
    Q_T_b = tl.dot(Q_T, b)

    # Solve R * x = Q^T * b
    x = tl.zeros((n, k), dtype=tl.float32)
    for i in range(n - 1, -1, -1):
        x[i, :] = (Q_T_b[i, :] - tl.sum(R[i, i+1:n] * x[i+1:n, :], axis=0)) / R[i, i]

    # Store the results
    tl.store(x_ptr + block_start * k + tl.arange(0, k), x)

import torch
import triton
import triton.language as tl

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]

    # Ensure the input tensors are on the same device
    device = A.device
    A = A.to(device)
    b = b.to(device)

    # Allocate memory for Q and R
    Q = torch.empty((m, n), device=device, dtype=A.dtype)
    R = torch.empty((n, n), device=device, dtype=A.dtype)

    # Perform QR decomposition
    grid = (1, 1, 1)
    BLOCK_SIZE = 128
    qr_decomposition_kernel[grid](A, R, Q, m, n, BLOCK_SIZE)

    # Allocate memory for the solution x
    x = torch.empty((n, k), device=device, dtype=A.dtype)

    # Solve the linear system
    solve_linear_system_kernel[grid](Q, R, b, x, m, n, k, BLOCK_SIZE)

    return x
