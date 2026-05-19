import triton
import triton.language as tl

@triton.jit
def pseudoinverse_svd_kernel(
    A_ptr,  # Input tensor of shape (*, m, n)
    U_ptr,  # Output tensor for U
    S_ptr,  # Output tensor for S
    Vh_ptr,  # Output tensor for Vh
    A_inv_ptr,  # Output tensor for the pseudoinverse of A
    m,  # Number of rows in A
    n,  # Number of columns in A
    rcond,  # Relative condition number threshold
    full_matrices,  # Whether to compute full SVD
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_size = tl.cdiv(A.shape[0], BLOCK_SIZE)
    batch_idx = pid // batch_size
    matrix_idx = pid % batch_size

    # Load the matrix A
    A = tl.load(A_ptr + batch_idx * m * n + matrix_idx * BLOCK_SIZE * BLOCK_SIZE, eviction_policy="evict_last")

    # Compute SVD
    U, S, Vh = tl.linalg.svd(A, full_matrices=full_matrices)

    # Compute the threshold for singular values
    max_S = tl.max(S, axis=0)
    threshold = rcond * max_S

    # Invert the singular values above the threshold
    S_inv = tl.where(S > threshold, 1.0 / S, 0.0)

    # Compute the pseudoinverse
    A_inv = tl.dot(tl.dot(Vh, tl.diag(S_inv)), U.T)

    # Store the results
    tl.store(U_ptr + batch_idx * m * m + matrix_idx * BLOCK_SIZE * BLOCK_SIZE, U)
    tl.store(S_ptr + batch_idx * min(m, n) + matrix_idx * BLOCK_SIZE, S)
    tl.store(Vh_ptr + batch_idx * n * n + matrix_idx * BLOCK_SIZE * BLOCK_SIZE, Vh)
    tl.store(A_inv_ptr + batch_idx * n * m + matrix_idx * BLOCK_SIZE * BLOCK_SIZE, A_inv)

import torch
import triton
import triton.language as tl

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> torch.Tensor:
    # Check input tensor properties
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert A.is_floating_point() or A.is_complex(), "Input tensor must be of float, double, cfloat, or cdouble dtype"
    assert A.is_contiguous(), "Input tensor must be contiguous"

    # Determine the shape of the input tensor
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2:]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Determine the shape of the output tensor
    if out is None:
        out = torch.empty(batch_shape + (n, m), dtype=A.dtype, device=A.device)

    # Allocate memory for U, S, and Vh
    U = torch.empty(batch_shape + (m, m), dtype=A.dtype, device=A.device)
    S = torch.empty(batch_shape + (min(m, n),), dtype=A.dtype, device=A.device)
    Vh = torch.empty(batch_shape + (n, n), dtype=A.dtype, device=A.device)

    # Launch the Triton kernel
    grid = (batch_size, 1, 1)
    block = (32, 32, 1)
    pseudoinverse_svd_kernel[grid, block](
        A, U, S, Vh, out, m, n, rcond, full_matrices
    )

    return out
