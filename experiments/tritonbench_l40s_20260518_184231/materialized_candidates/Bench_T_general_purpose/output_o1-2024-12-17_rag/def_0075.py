import math
import torch
import triton
import triton.language as tl

# Kernel to update the k-th column of the Cholesky factor L.
# For each row i > k, computes:
# L[i, k] = ( A[i, k] - sum_{j=0..k-1}( L[i, j] * L[k, j] ) ) / L[k, k]
@triton.jit
def cholesky_update_col_kernel(
    A_ptr,       # pointer to A
    L_ptr,       # pointer to L
    n,           # matrix dimension
    k,           # current column being updated
    BLOCK_SIZE: tl.constexpr
):
    # block offset for threads
    block_id = tl.program_id(0)
    row_start = block_id * BLOCK_SIZE + (k + 1)
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # For valid rows, compute partial sums for the k-th column
    # sum_{j=0..k-1}(L[i,j]*L[k,j])
    # then compute L[i,k]
    i = offsets
    # Initialize partial_sum with 0.0
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for j in range(k):
        Lij = tl.load(L_ptr + i * n + j, mask=mask)
        Lkj = tl.load(L_ptr + k * n + j)  # k-th row doesn't need mask
