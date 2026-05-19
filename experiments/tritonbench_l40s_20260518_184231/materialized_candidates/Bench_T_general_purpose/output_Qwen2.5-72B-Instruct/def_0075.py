import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A, L, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row = block_start + tl.arange(0, BLOCK_SIZE)
    col = block_start + tl.arange(0, BLOCK_SIZE)

    mask = (row < n) & (col < n)
    row = row[mask]
    col = col[mask]

    L_block = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    A_block = tl.load(A + row[:, None] * n + col[None, :], mask=mask, other=0.0)

    for i in range(block_start):
        L_i = tl.load(L + i * n + row, mask=row < n, other=0.0)
        L_ii = tl.load(L + i * n + i, mask=i < n, other=0.0)
        L_block += L_i[:, None] * L_ii[None, :]
    
    L_block = A_block - L_block
    L_block = tl.where(row[:, None] >= col[None, :], L_block, 0.0)
    L_block = tl.where(row == col, tl.sqrt(L_block), L_block / tl.sqrt(L_block[tl.arange(BLOCK_SIZE), tl.arange(BLOCK_SIZE)]))

    tl.store(L + row[:, None] * n + col[None, :], L_block, mask=mask)

@triton.jit
def solve_kernel(L, b, x, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    row = block_start + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, k)

    mask = (row < n)
    row = row[mask]

    x_block = tl.zeros((BLOCK_SIZE, k), dtype=tl.float32)
    b_block = tl.load(b + row[:, None] * k + col[None, :], mask=mask, other=0.0)

    for i in range(block_start):
        L_i = tl.load(L + i * n + row, mask=row < n, other=0.0)
        x_i = tl.load(x + i * k + col, mask=i < n, other=0.0)
        b_block -= L_i[:, None] * x_i[None, :]

    x_block = b_block / tl.load(L + row[:, None] * n + row[None, :], mask=mask, other=0.0)
    tl.store(x + row[:, None] * k + col[None, :], x_block, mask=mask)

import torch
import triton

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n, k = b.shape
    assert A.shape == (n, n), "Matrix A must be of shape (n, n)"
    assert b.shape == (n, k), "Tensor b must be of shape (n, k)"

    # Allocate memory for the Cholesky factor L
    L = torch.zeros_like(A, device=A.device)

    # Cholesky decomposition
    BLOCK_SIZE = 16
    grid = (n // BLOCK_SIZE + (n % BLOCK_SIZE > 0),)
    cholesky_kernel[grid](A, L, n, BLOCK_SIZE)

    # Allocate memory for the solution x
    x = torch.zeros_like(b, device=b.device)

    # Solve the linear system
    grid = (n // BLOCK_SIZE + (n % BLOCK_SIZE > 0),)
    solve_kernel[grid](L, b, x, n, k, BLOCK_SIZE)

    return x

# Example usage
A = torch.tensor([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=torch.float32, device='cuda')
b = torch.tensor([[1], [2], [3]], dtype=torch.float32, device='cuda')

x = fused_cholesky_solve(A, b)
print(x)
