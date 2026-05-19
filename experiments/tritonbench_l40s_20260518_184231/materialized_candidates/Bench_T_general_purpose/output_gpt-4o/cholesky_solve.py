import triton
import triton.language as tl

@triton.jit
def cholesky_solve_kernel(B_ptr, L_ptr, X_ptr, n, k, upper, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Determine the batch index and the block start
    batch_idx = pid // (n // BLOCK_SIZE)
    row_start = (pid % (n // BLOCK_SIZE)) * BLOCK_SIZE

    # Pointers to the current batch
    B = B_ptr + batch_idx * n * k
    L = L_ptr + batch_idx * n * n
    X = X_ptr + batch_idx * n * k

    # Load the block of B and L
    B_block = tl.load(B + row_start * k + tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, k)[None, :])
    L_block = tl.load(L + row_start * n + tl.arange(0, BLOCK_SIZE)[:, None] * n + tl.arange(0, BLOCK_SIZE)[None, :])

    # Initialize X_block
    X_block = tl.zeros([BLOCK_SIZE, k], dtype=B_block.dtype)

    # Forward or backward substitution
    if upper:
        # Solve L^H Y = B
        for i in range(BLOCK_SIZE):
            for j in range(k):
                X_block[i, j] = (B_block[i, j] - tl.dot(X_block[:i, j], L_block[:i, i])) / L_block[i, i]
        # Solve L X = Y
        for i in range(BLOCK_SIZE-1, -1, -1):
            for j in range(k):
                X_block[i, j] = (X_block[i, j] - tl.dot(X_block[i+1:, j], L_block[i+1:, i])) / L_block[i, i]
    else:
        # Solve L Y = B
        for i in range(BLOCK_SIZE):
            for j in range(k):
                X_block[i, j] = (B_block[i, j] - tl.dot(X_block[:i, j], L_block[i, :i])) / L_block[i, i]
        # Solve L^H X = Y
        for i in range(BLOCK_SIZE-1, -1, -1):
            for j in range(k):
                X_block[i, j] = (X_block[i, j] - tl.dot(X_block[i+1:, j], L_block[i, i+1:])) / L_block[i, i]

    # Store the result in X
    tl.store(X + row_start * k + tl.arange(0, BLOCK_SIZE)[:, None] * k + tl.arange(0, k)[None, :], X_block)

import torch

def cholesky_solve(B, L, upper=False, *, out=None):
    assert B.shape[:-2] == L.shape[:-2], "Batch dimensions must match"
    assert B.shape[-2] == L.shape[-1], "Matrix dimensions must match"

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(B)

    # Get the number of batches and the size of the matrices
    *batch_dims, n, k = B.shape

    # Flatten batch dimensions for Triton
    batch_size = int(torch.prod(torch.tensor(batch_dims)))

    # Launch Triton kernel
    grid = (batch_size * (n // 32),)
    cholesky_solve_kernel[grid](B, L, out, n, k, upper, BLOCK_SIZE=32)

    return out
