import torch
import triton
import triton.language as tl

# Triton kernel to perform the low-rank SVD approximation
@triton.jit
def low_rank_svd_kernel(U, S, Vh, A, k, m, n, BLOCK_SIZE: tl.constexpr):
    # Compute the rank-k approximation using the truncated SVD components
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    
    mask_row = row < m
    mask_col = col < n

    # Load U, S, Vh into shared memory
    Uk = tl.load(U + row[:, None] * k + tl.arange(0, k), mask=mask_row[:, None])
    Sk = tl.load(S + tl.arange(0, k), mask=tl.arange(0, k) < k)
    Vhk = tl.load(Vh + tl.arange(0, k) * n + col[None, :], mask=mask_col[None, :])

    # Compute Ak = Uk @ diag(Sk) @ Vhk
    Ak = tl.dot(Uk * Sk[None, :], Vhk)

    # Store result in output tensor
    tl.store(A + row[:, None] * n + col[None, :], Ak, mask=mask_row[:, None] & mask_col[None, :])

def low_rank_svd_approximation(A, k, *, full_matrices=True, out=None):
    # Check input dimensions
    assert A.ndim >= 2, "Input tensor must have at least 2 dimensions"
    m, n = A.shape[-2], A.shape[-1]
    assert 1 <= k <= min(m, n), "k must satisfy 1 <= k <= min(m, n)"
    
    # Perform SVD using PyTorch (can be accelerated by Triton in the future)
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Slice to get top-k components
    Uk = U[..., :k]
    Sk = S[..., :k]
    Vhk = Vh[..., :k, :]

    # Prepare output tensor
    if out is None:
        out = torch.empty_like(A)

    # Get batch dimensions
    batch_dims = A.shape[:-2]
    num_batches = torch.prod(torch.tensor(batch_dims)).item()

    # Launch Triton kernel for each batch
    for batch_idx in range(num_batches):
        # Flatten batch dimensions
        batch_offset = batch_idx * m * n
        # Call the Triton kernel
        low_rank_svd_kernel[(num_batches,)](
            Uk.view(-1, m, k)[batch_idx],
            Sk.view(-1, k)[batch_idx],
            Vhk.view(-1, k, n)[batch_idx],
            out.view(-1, m, n)[batch_idx],
            k,
            m,
            n,
            BLOCK_SIZE=128  # Define an appropriate block size
        )

    return out
