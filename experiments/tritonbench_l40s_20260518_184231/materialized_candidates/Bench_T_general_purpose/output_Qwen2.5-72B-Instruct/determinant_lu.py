import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A,  # Input matrix of shape (*, n, n)
    P,  # Permutation matrix (if pivot=True)
    L,  # Lower triangular matrix
    U,  # Upper triangular matrix
    n,  # Size of the square matrix
    pivot,  # Whether to use pivoting
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Loop over the blocks of the matrix
    for i in range(n):
        # Compute the pivot row
        if pivot:
            max_idx = tl.argmax(tl.abs(A[block_start + i, block_start + i : block_start + i + BLOCK_SIZE]), axis=0)
            P[block_start + i, block_start + i + max_idx] = 1.0
            A[block_start + i, block_start + i : block_start + i + BLOCK_SIZE] = A[block_start + i + max_idx, block_start + i : block_start + i + BLOCK_SIZE]

        # Compute the L and U matrices
        for j in range(i + 1, n):
            L[block_start + j, block_start + i] = A[block_start + j, block_start + i] / A[block_start + i, block_start + i]
            A[block_start + j, block_start + i : block_start + j] -= L[block_start + j, block_start + i] * A[block_start + i, block_start + i : block_start + j]

        U[block_start + i, block_start + i : block_start + i + BLOCK_SIZE] = A[block_start + i, block_start + i : block_start + i + BLOCK_SIZE]

@triton.jit
def determinant_kernel(
    U,  # Upper triangular matrix
    det,  # Output determinant
    n,  # Size of the square matrix
    pivot,  # Whether to use pivoting
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Initialize the determinant
    det_val = 1.0

    # Compute the determinant
    for i in range(n):
        det_val *= U[block_start + i, block_start + i]

    # Adjust by the sign of the permutation matrix if pivoting is used
    if pivot:
        det_val *= tl.sign(tl.sum(tl.diag(P[block_start : block_start + n, block_start : block_start + n])))

    det[block_start] = det_val

import torch
import triton
import triton.language as tl

def determinant_lu(A, *, pivot=True, out=None):
    # Check input tensor shape and dtype
    if A.dim() < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Input tensor must be a square matrix or a batch of square matrices.")
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("Input tensor must have dtype float32, float64, complex64, or complex128.")

    # Get the size of the square matrix
    n = A.shape[-1]

    # Initialize the permutation matrix P, lower triangular matrix L, and upper triangular matrix U
    P = torch.eye(n, device=A.device, dtype=A.dtype) if pivot else None
    L = torch.zeros_like(A)
    U = torch.zeros_like(A)

    # Perform LU decomposition
    grid = (A.shape[0] * A.shape[1] // 1024 + 1,)
    lu_decomposition_kernel[grid](A, P, L, U, n, pivot, BLOCK_SIZE=1024)

    # Initialize the output tensor for the determinant
    if out is None:
        out = torch.empty(A.shape[:-2], device=A.device, dtype=A.dtype)

    # Compute the determinant
    determinant_kernel[grid](U, out, n, pivot, BLOCK_SIZE=1024)

    return out
