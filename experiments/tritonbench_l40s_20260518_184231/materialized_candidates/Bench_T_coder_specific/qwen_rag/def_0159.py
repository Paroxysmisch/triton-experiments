import triton
import triton.language as tl
import torch

# Triton kernel to perform the Cholesky decomposition
@triton.jit
def cholesky_kernel(
    A,  # Pointer to the input matrix
    L,  # Pointer to the output matrix
    M,  # Number of rows/columns in the input matrix
    upper,  # Whether to compute the upper triangular matrix
    M_BLOCK_SIZE: tl.constexpr,  # Block size for rows/columns
):
    pid = tl.program_id(0)
    row = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)
    col = pid * M_BLOCK_SIZE + tl.arange(0, M_BLOCK_SIZE)
    row_mask = row < M
    col_mask = col < M
    mask = row_mask & col_mask

    x = tl.load(A + row * M + col, mask, other=0.0)
    tl.store(L + row * M + col, x, mask=mask)

    for k in range(pid, M, M_BLOCK_SIZE):
        k_mask = k < M
        if k >= pid:
            diag_kk = tl.load(L + k * M + k, mask=k_mask, other=0.0)
            x = tl.where(col == k, x / diag_kk, x)
            tl.store(L + row * M + col, x, mask=mask)

        for j in range(k + 1, M, M_BLOCK_SIZE):
            j_mask = j < M
            if j >= pid:
                diag_jj = tl.load(L + j * M + j, mask=j_mask, other=0.0)
                x = tl.where(col == j, x / diag_jj, x)
                tl.store(L + row * M + col, x, mask=mask)

                if upper:
                    y = tl.load(L + row * M + j, mask=mask, other=0.0)
                    x = tl.where(row == j, x - y * y, x)
                else:
                    y = tl.load(L + j * M + row, mask=mask, other=0.0)
                    x = tl.where(col == j, x - y * y, x)

                tl.store(L + row * M + col, x, mask=mask)

# Function to compute the Cholesky decomposition of a matrix or batch of matrices
def linalg_cholesky(A, upper=False, out=None):
    A = A.contiguous()
    if out is None:
        out = torch.empty_like(A)
    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    M, N = A.shape[-2:]
    assert M == N, "Input matrix must be square"

    with torch.cuda.device(A.device):
        if len(A.shape) == 2:
            grid = lambda meta: (triton.cdiv(M, meta["M_BLOCK_SIZE"]),)
            cholesky_kernel[grid](A, out, M, upper, M_BLOCK_SIZE=32)
        else:
            batch = int(torch.numel(A) / M / N)
            B = A.view(batch, -1)
            C = out.view(batch, -1)
            grid = lambda meta: (triton.cdiv(batch, meta["BATCH_BLOCK_SIZE"]),)
            cholesky_kernel[grid](B, C, M, upper, BATCH_BLOCK_SIZE=32)
            C = C.view(A.shape)
            out = C

    return out
