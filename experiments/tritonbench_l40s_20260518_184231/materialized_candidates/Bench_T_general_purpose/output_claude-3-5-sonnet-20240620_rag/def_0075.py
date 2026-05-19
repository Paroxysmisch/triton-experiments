import torch
import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = row[:, None] + tl.arange(0, BLOCK_SIZE)[None, :]

    mask = cols < n
    row_mask = row < n

    A = tl.load(A_ptr + row[:, None] * n + cols, mask=mask)

    for k in range(0, n):
        k_val = k
        if k < BLOCK_SIZE:
            if row_mask:
                if k_val < cols:
                    l_ik = tl.load(L_ptr + row * n + k_val, mask=row_mask)
                    A = A - l_ik[:, None] * l_ik[None, :]
                if k_val == cols:
                    A = tl.sqrt(A)
                    tl.store(L_ptr + row * n + k_val, A, mask=row_mask)

        else:
            l_ik = tl.load(L_ptr + row[:, None] * n + k_val, mask=row_mask)
            A = A - l_ik * l_ik


@triton.jit
def solve_kernel(L_ptr, b_ptr, x_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, BLOCK_SIZE)
    mask = row < n
    L = tl.load(L_ptr + row[:, None] * n + row, mask=mask)
    b = tl.load(b_ptr + row[:, None] * k + col[None, :], mask=mask[:,None])

    y = b / L[:, None]
    tl.store(x_ptr + row[:, None] * k + col[None, :], y, mask=mask[:,None])


def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    k = b.shape[1]

    L = torch.zeros_like(A)
    x = torch.zeros_like(b)

    BLOCK_SIZE = 128
    grid_size = (triton.cdiv(n, BLOCK_SIZE),)

    cholesky_kernel[grid_size](A, L, n, BLOCK_SIZE)

    grid_size = (triton.cdiv(n, BLOCK_SIZE),)
    solve_kernel[grid_size](L, b, x, n, k, BLOCK_SIZE)

    return x
