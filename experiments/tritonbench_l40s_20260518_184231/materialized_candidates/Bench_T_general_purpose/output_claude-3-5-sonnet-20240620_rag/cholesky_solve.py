import triton
import triton.language as tl
import torch

# Triton kernel to solve the system of linear equations using Cholesky decomposition
@triton.jit
def cholesky_solve_kernel(B, L, out, n, k, upper, batch_size, B_BATCH_SIZE: tl.constexpr):
    batch_id = tl.program_id(0)
    row = tl.arange(0, B_BATCH_SIZE) + batch_id * B_BATCH_SIZE
    mask = row < batch_size

    for i in range(n):
        if upper:
            # Forward substitution for upper triangular matrix
            for j in range(i + 1, n):
                B[row, j] -= L[row, i] * B[row, j] / L[row, i, i]
        else:
            # Forward substitution for lower triangular matrix
            for j in range(i):
                B[row, j] -= L[row, i] * B[row, j] / L[row, i, i]

    # Back substitution
    for i in range(n - 1, -1, -1):
        if upper:
            B[row, i] /= L[row, i, i]
            for j in range(i):
                B[row, j] -= L[row, i] * B[row, j] / L[row, i, i]
        else:
            B[row, i] /= L[row, i, i]
            for j in range(i + 1, n):
                B[row, j] -= L[row, i] * B[row, j] / L[row, i, i]

# Wrapper function for cholesky_solve
def cholesky_solve(B, L, upper=False, *, out=None):
    B = B.contiguous()
    L = L.contiguous()
    if out is None:
        out = torch.empty_like(B)

    batch_size, n, k = B.shape
    B_BATCH_SIZE = 32  # Define a suitable batch size

    grid = lambda meta: (triton.cdiv(batch_size, meta['B_BATCH_SIZE']),)
    cholesky_solve_kernel[grid](B, L, out, n, k, upper, batch_size, B_BATCH_SIZE=B_BATCH_SIZE)

    return out
