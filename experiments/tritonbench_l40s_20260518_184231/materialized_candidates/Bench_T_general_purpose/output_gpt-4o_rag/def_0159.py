import triton
import triton.language as tl
import torch

# Triton kernel for Cholesky decomposition
@triton.jit
def cholesky_kernel(
    A_ptr,  # Pointer to the input matrix
    L_ptr,  # Pointer to the output matrix (lower triangular)
    N,  # Number of rows and columns in the input matrix
    stride,  # Stride for batch processing
    BLOCK_SIZE: tl.constexpr  # Block size for the kernel
):
    pid = tl.program_id(0)
    offset = pid * stride

    # Load the block of the matrix
    for i in range(0, N, BLOCK_SIZE):
        row_idx = i + tl.arange(0, BLOCK_SIZE)
        col_idx = i + tl.arange(0, BLOCK_SIZE)
        row_mask = row_idx < N
        col_mask = col_idx < N

        # Load a block of A
        A = tl.load(A_ptr + offset + row_idx[:, None] * N + col_idx[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0)

        # Perform Cholesky decomposition on the block
        for j in range(BLOCK_SIZE):
            if row_idx[j] < N:
                A[j, j] = tl.sqrt(A[j, j])
                for k in range(j + 1, BLOCK_SIZE):
                    if row_idx[k] < N:
                        A[k, j] = A[k, j] / A[j, j]
                        A[k, k:] = A[k, k:] - A[k, j] * A[j, k:]

        # Store the lower triangular block
        tl.store(L_ptr + offset + row_idx[:, None] * N + col_idx[None, :], A, mask=row_mask[:, None] & col_mask[None, :])

# Wrapper function for Cholesky decomposition
def cholesky(A, upper=False, out=None):
    assert A.shape[-1] == A.shape[-2], "Matrix must be square"
    N = A.shape[-1]
    batch_size = A.shape[0] if len(A.shape) > 2 else 1
    A = A.contiguous()

    if out is None:
        out = torch.empty_like(A)

    grid = lambda meta: (batch_size,)
    cholesky_kernel[grid](
        A,
        out,
        N,
        N * N,
        BLOCK_SIZE=32
    )

    if upper:
        out = out.transpose(-1, -2).conj()

    return out

# Example usage
A = torch.randn(4, 4, dtype=torch.float32, device='cuda')
A = A @ A.transpose(-1, -2)  # Make it positive definite
L = cholesky(A)
