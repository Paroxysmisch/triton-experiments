import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n, stride, upper, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row = tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    A = tl.load(A_ptr + pid * stride + row[:, None] * n + col[None, :], mask=(row < n) & (col < n), other=0.0)
    
    # Cholesky decomposition
    for k in range(0, n):
        # Update diagonal
        A[k, k] = tl.sqrt(A[k, k])
        # Update below diagonal
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]
        # Update remaining submatrix
        for i in range(k + 1, n):
            for j in range(k + 1, i + 1):
                A[i, j] = A[i, j] - A[i, k] * tl.conj(A[j, k])

    # Store result
    if upper:
        L = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=A.dtype)
        L = tl.where(row[:, None] <= col[None, :], A, L)
    else:
        L = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=A.dtype)
        L = tl.where(row[:, None] >= col[None, :], A, L)

    tl.store(L_ptr + pid * stride + row[:, None] * n + col[None, :], L, mask=(row < n) & (col < n))

def cholesky(A, *, upper=False, out=None):
    # Ensure input is a tensor
    if not isinstance(A, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Check input dimensions
    if A.dim() < 2 or A.size(-1) != A.size(-2):
        raise ValueError("Input must be a batch of square matrices")

    n = A.size(-1)
    batch = A.numel() // (n * n)

    # Allocate output if necessary
    if out is None:
        out = torch.empty_like(A)

    # Move data to GPU
    A = A.to(device='cuda')
    out = out.to(device='cuda')

    # Define grid and block size
    BLOCK_SIZE = 16  # This can be tuned
    grid = (batch,)

    # Launch kernel
    cholesky_kernel[grid](A, out, n, A.stride(0), upper, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
A = torch.randn(2, 3, 3, dtype=torch.float32, device='cuda')
A = A @ A.transpose(-1, -2) + 1e-3 * torch.eye(3, device='cuda')  # Make it positive definite
L = cholesky(A, upper=False)
