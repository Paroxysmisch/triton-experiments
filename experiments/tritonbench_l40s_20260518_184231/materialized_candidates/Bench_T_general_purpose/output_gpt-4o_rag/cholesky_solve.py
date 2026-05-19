import triton
import triton.language as tl
import torch

# Triton kernel for forward and backward substitution
@triton.jit
def cholesky_solve_kernel(
    B,  # Pointer to the right-hand side matrix
    L,  # Pointer to the Cholesky factor matrix
    X,  # Pointer to the output matrix
    M,  # Number of rows in L
    N,  # Number of columns in B
    upper: tl.constexpr,  # Flag for upper or lower triangular
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = row < M

    # Forward or backward substitution
    for k in range(M):
        k_mask = row == k
        b_k = tl.load(B + k * N, mask=k_mask)
        if upper:
            # Backward substitution for upper triangular
            l_kk = tl.load(L + k * M + k, mask=k_mask)
            x_k = b_k / l_kk
            tl.store(X + k * N, x_k, mask=k_mask)
            b_k = tl.load(B + row * N, mask=row_mask)
            l_ik = tl.load(L + row * M + k, mask=row_mask)
            b_k -= l_ik * x_k
            tl.store(B + row * N, b_k, mask=row_mask)
        else:
            # Forward substitution for lower triangular
            l_kk = tl.load(L + k * M + k, mask=k_mask)
            x_k = b_k / l_kk
            tl.store(X + k * N, x_k, mask=k_mask)
            b_k = tl.load(B + row * N, mask=row_mask)
            l_ki = tl.load(L + k * M + row, mask=row_mask)
            b_k -= l_ki * x_k
            tl.store(B + row * N, b_k, mask=row_mask)

# Wrapper function
def cholesky_solve(B, L, upper=False, *, out=None):
    B = B.contiguous()
    L = L.contiguous()
    M, N = L.shape[-2], B.shape[-1]
    if out is None:
        out = torch.empty_like(B)

    assert L.shape[-1] == M, "L must be square"
    assert B.shape[-2] == M, "B must have the same number of rows as L"

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)

    cholesky_solve_kernel[grid](
        B, L, out, M, N, upper,
        BLOCK_SIZE=32,
    )

    return out

# Example usage
B = torch.randn(4, 4, device='cuda', dtype=torch.float32)
L = torch.linalg.cholesky(torch.eye(4, device='cuda', dtype=torch.float32) + 0.1 * torch.randn(4, 4, device='cuda', dtype=torch.float32))
X = cholesky_solve(B, L)
