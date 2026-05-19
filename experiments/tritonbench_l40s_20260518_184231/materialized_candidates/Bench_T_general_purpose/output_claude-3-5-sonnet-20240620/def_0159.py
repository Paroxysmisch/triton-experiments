import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n, batch_size, upper):
    # Define grid size
    row = tl.program_id(0)
    col = tl.program_id(1)
    batch = tl.program_id(2)

    # Calculate the global index
    idx = row * n + col

    # Load the matrix A
    A = tl.load(A_ptr + (batch * n * n + row * n + col) * 4)  # Assuming float32 for simplicity

    # Initialize L
    L = tl.zeros((n, n), dtype=tl.float32)

    # Cholesky decomposition logic
    if row >= col:
        sum = tl.zeros((), dtype=tl.float32)
        for k in range(col):
            sum += L[row, k] * L[col, k]
        L[row, col] = (A - sum) / L[col, col] if row != col else tl.sqrt(A)

    # Store the result
    tl.store(L_ptr + (batch * n * n + row * n + col) * 4, L[row, col])

import torch
from torch import Tensor
import triton

def cholesky(A: Tensor, *, upper: bool = False, out: Tensor = None) -> Tensor:
    # Validate input
    if not (A.is_complex() or A.is_floating_point()):
        raise TypeError("Input tensor must be of float or complex type.")
    
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError("Input tensor must be at least 2D and square in the last two dimensions.")

    # Get dimensions
    batch_size, n = A.shape[:-2], A.shape[-2]

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(A)

    # Launch Triton kernel
    grid = (n, n, batch_size)
    cholesky_kernel[grid](A, out, n, batch_size, upper)

    # Handle upper triangular output
    if upper:
        out = out.transpose(-1, -2).conj()

    return out
