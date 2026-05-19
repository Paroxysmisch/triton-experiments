import triton
import triton.language as tl

@triton.jit
def lu_decomposition_kernel(
    A_ptr, P_ptr, L_ptr, U_ptr, M, N, pivot, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load the matrix block
    A = tl.load(A_ptr + block_start * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < M, other=0.0)

    # Initialize P, L, and U
    P = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.int32)
    L = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=A.dtype)
    U = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=A.dtype)

    for i in range(BLOCK_SIZE):
        if pivot:
            # Find the pivot
            max_idx = tl.argmax(tl.abs(A[i:, i]), axis=0) + i
            P[i, max_idx] = 1
            # Swap rows in A
            temp = A[i, i:]
            A[i, i:] = A[max_idx, i:]
            A[max_idx, i:] = temp

        # Compute the pivot row in U
        U[i, i:] = A[i, i:]

        # Compute the pivot column in L
        if i < BLOCK_SIZE - 1:
            L[i + 1:, i] = A[i + 1:, i] / U[i, i]

        # Update the trailing submatrix
        A[i + 1:, i + 1:] -= tl.outer(L[i + 1:, i], U[i, i + 1:])

    # Store the results
    tl.store(P_ptr + block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), P, mask=block_start + tl.arange(0, BLOCK_SIZE) < M)
    tl.store(L_ptr + block_start * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, BLOCK_SIZE), L, mask=block_start + tl.arange(0, BLOCK_SIZE) < M)
    tl.store(U_ptr + block_start * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, BLOCK_SIZE), U, mask=block_start + tl.arange(0, BLOCK_SIZE) < M)

import torch
import triton
import triton.language as tl

def lu(A, *, pivot=True, out=None):
    # Check input tensor
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")
    if A.dim() < 2:
        raise ValueError("A must have at least 2 dimensions")
    if A.dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        raise ValueError("A must have dtype float, double, cfloat, or cdouble")

    # Get dimensions
    *batch_dims, M, N = A.shape
    batch_size = 1 if not batch_dims else batch_dims[0]
    device = A.device

    # Initialize output tensors
    if out is None:
        P = torch.zeros((*batch_dims, M, M), dtype=torch.int32, device=device)
        L = torch.zeros_like(A)
        U = torch.zeros_like(A)
    else:
        P, L, U = out
        if P.shape != (*batch_dims, M, M) or P.dtype != torch.int32 or P.device != device:
            raise ValueError("P must have shape (*, M, M), dtype int32, and be on the same device as A")
        if L.shape != A.shape or L.dtype != A.dtype or L.device != device:
            raise ValueError("L must have the same shape, dtype, and device as A")
        if U.shape != A.shape or U.dtype != A.dtype or U.device != device:
            raise ValueError("U must have the same shape, dtype, and device as A")

    # Launch the Triton kernel
    grid = (batch_size * (M // 32),)
    lu_decomposition_kernel[grid](
        A, P, L, U, M, N, pivot, BLOCK_SIZE=32
    )

    return P, L, U
