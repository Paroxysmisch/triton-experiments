import triton
import triton.language as tl
import torch

# Triton kernel to compute the Cholesky decomposition of a single matrix
@triton.jit
def cholesky_kernel(
    A,  # Pointer to the input matrix
    L,  # Pointer to the output matrix
    n,  # Size of the matrix
    upper: tl.constexpr,  # Whether to compute the upper triangular matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for the matrix
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Compute the row and column indices for the current block
    row = block_start + tl.arange(0, BLOCK_SIZE)[:, None]
    col = block_start + tl.arange(0, BLOCK_SIZE)[None, :]

    # Compute the mask for valid indices
    mask = (row < n) & (col < n)

    # Load the block from the input matrix
    a_block = tl.load(A + row * n + col, mask=mask, other=0.0)

    # Compute the Cholesky decomposition for the current block
    for k in range(block_start, block_start + BLOCK_SIZE):
        if k < n:
            # Compute the diagonal element
            if upper:
                a_ii = tl.sqrt(a_block[k - block_start, k - block_start])
                a_block[k - block_start, k - block_start] = a_ii
            else:
                a_ii = tl.sqrt(a_block[k - block_start, k - block_start])
                a_block[k - block_start, k - block_start] = a_ii

            # Compute the off-diagonal elements
            for j in range(k + 1, block_start + BLOCK_SIZE):
                if j < n:
                    if upper:
                        a_ij = (a_block[k - block_start, j - block_start] - tl.sum(a_block[k - block_start, :k - block_start] * a_block[j - block_start, :k - block_start])) / a_ii
                        a_block[k - block_start, j - block_start] = a_ij
                    else:
                        a_ij = (a_block[j - block_start, k - block_start] - tl.sum(a_block[j - block_start, :k - block_start] * a_block[k - block_start, :k - block_start])) / a_ii
                        a_block[j - block_start, k - block_start] = a_ij

    # Store the result in the output matrix
    if upper:
        tl.store(L + col * n + row, a_block, mask=mask)
    else:
        tl.store(L + row * n + col, a_block, mask=mask)

# Triton kernel to compute the Cholesky decomposition for batched matrices
@triton.jit
def cholesky_batch_kernel(
    A,  # Pointer to the input matrix batch
    L,  # Pointer to the output matrix batch
    batch,  # Number of matrices in the batch
    n,  # Size of the matrix
    upper: tl.constexpr,  # Whether to compute the upper triangular matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for the matrix
):
    pid = tl.program_id(0)
    batch_id = pid // (n // BLOCK_SIZE)
    block_start = (pid % (n // BLOCK_SIZE)) * BLOCK_SIZE

    # Compute the row and column indices for the current block
    row = block_start + tl.arange(0, BLOCK_SIZE)[:, None]
    col = block_start + tl.arange(0, BLOCK_SIZE)[None, :]

    # Compute the mask for valid indices
    mask = (row < n) & (col < n)

    # Load the block from the input matrix
    a_block = tl.load(A + batch_id * n * n + row * n + col, mask=mask, other=0.0)

    # Compute the Cholesky decomposition for the current block
    for k in range(block_start, block_start + BLOCK_SIZE):
        if k < n:
            # Compute the diagonal element
            if upper:
                a_ii = tl.sqrt(a_block[k - block_start, k - block_start])
                a_block[k - block_start, k - block_start] = a_ii
            else:
                a_ii = tl.sqrt(a_block[k - block_start, k - block_start])
                a_block[k - block_start, k - block_start] = a_ii

            # Compute the off-diagonal elements
            for j in range(k + 1, block_start + BLOCK_SIZE):
                if j < n:
                    if upper:
                        a_ij = (a_block[k - block_start, j - block_start] - tl.sum(a_block[k - block_start, :k - block_start] * a_block[j - block_start, :k - block_start])) / a_ii
                        a_block[k - block_start, j - block_start] = a_ij
                    else:
                        a_ij = (a_block[j - block_start, k - block_start] - tl.sum(a_block[j - block_start, :k - block_start] * a_block[k - block_start, :k - block_start])) / a_ii
                        a_block[j - block_start, k - block_start] = a_ij

    # Store the result in the output matrix
    if upper:
        tl.store(L + batch_id * n * n + col * n + row, a_block, mask=mask)
    else:
        tl.store(L + batch_id * n * n + row * n + col, a_block, mask=mask)

# Wrapper function to compute the Cholesky decomposition
def linalg_cholesky(A, *, upper=False, out=None):
    A = A.contiguous()
    if out is None:
        out = torch.empty_like(A)
    else:
        assert out.shape == A.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == A.dtype, "Output tensor must have the same dtype as input tensor"

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    n = A.shape[-1]
    assert A.shape[-2] == n, "Input tensor must be a square matrix or batch of square matrices"

    with torch.cuda.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            cholesky_kernel[grid](A, out, n, upper, BLOCK_SIZE=32)
        else:
            batch = A.shape[:-2].numel()
            grid = lambda meta: (triton.cdiv(batch * n, meta["BLOCK_SIZE"]),)
            cholesky_batch_kernel[grid](A, out, batch, n, upper, BLOCK_SIZE=32)

    return out

# Example usage
A = torch.randn(3, 3, 3, 3, device='cuda', dtype=torch.float64)
A = A @ A.transpose(-2, -1)  # Ensure A is symmetric positive-definite
L = linalg_cholesky(A, upper=False)
print(L)
