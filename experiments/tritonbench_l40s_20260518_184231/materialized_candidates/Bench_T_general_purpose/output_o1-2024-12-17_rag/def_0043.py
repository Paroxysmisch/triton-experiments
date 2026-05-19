import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------------
# Kernel: Symmetric Matrix-Vector Multiply with alpha/beta scaling
#
# Computes: y[i] = alpha * sum_j(A[i, j] * x[j]) + beta * y[i]
# Assumes A is symmetric. For simplicity, we do a standard mat-vec multiply ignoring
# additional symmetry-specific optimizations.
# 
# Dimensions:
#   - A is n x n
#   - x, y are n x 1 (stored as 1D)
#
# We'll launch one block per row of A; each block iterates over columns in chunks.
# ------------------------------------------------------------------------------------
@triton.jit
def _sym_mv_kernel(
    A_ptr,    # pointer to A (n x n)
    x_ptr,    # pointer to x (n
