import triton
import triton.language as tl

@triton.jit
def svd_kernel(A_ptr, U_ptr, S_ptr, V_ptr, Ak_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_size = A_ptr.shape[0]
    batch_idx = pid // (M * N)
    row_idx = (pid % (M * N)) // N
    col_idx = (pid % (M * N)) % N

    # Load the matrix A
    A = tl.load(A_ptr + batch_idx * M * N + row_idx * N + col_idx)

    # Perform SVD (this is a placeholder for the actual SVD computation)
    U, S, V = tl.linalg.svd(A)

    # Select the top-k singular values and vectors
    U_k = U[:, :K]
    S_k = S[:K]
    V_k = V[:, :K]

    # Construct the rank-k approximation
    Ak = U_k @ tl.diag(S_k) @ V_k.T

    # Store the result
    tl.store(Ak_ptr + batch_idx * M * N + row_idx * N + col_idx, Ak)

import torch
import triton
import triton.language as tl

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    # Check input dimensions
    if A.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    m, n = A.shape[-2:]
    if k < 1 or k > min(m, n):
        raise ValueError(f"Rank k must satisfy 1 <= k <= min(m, n), but got k={k}.")

    # Compute SVD
    U, S, V = torch.linalg.svd(A, full_matrices=full_matrices)

    # Select the top-k singular values and vectors
    U_k = U[..., :k]
    S_k = S[..., :k]
    V_k = V[..., :k]

    # Construct the rank-k approximation
    Ak = U_k @ torch.diag_embed(S_k) @ V_k.transpose(-2, -1)

    # Handle the output tensor
    if out is not None:
        if out.shape != A.shape:
            raise ValueError(f"Output tensor shape must match input tensor shape, but got {out.shape} and {A.shape}.")
        out.copy_(Ak)
        return out
    else:
        return Ak
