import torch
import triton
import triton.language as tl
from triton.language.extra import householder

@triton.jit
def _lu_solve(L, U, b):
    n = L.shape[0]
    # Solve Ly = b
    y = tl.zeros((n,), dtype=tl.float32)
    for i in range(n):
        idx = i + tl.arange(0, n)  # idx: [i, i+1, ..., n-1]
        mask = idx >= i
        y += tl.where(mask, -tl.sum(L[i:, i] * y[idx[mask]], axis=0) / L[i, i], 0.0)
    y += tl.where(tl.arange(0, n) >= n - 1, b / L[n - 1, n - 1], 0.0)

    # Solve Ux = y
    x = tl.zeros((n,), dtype=tl.float32)
    for i in range(n - 1, -1, -1):
        idx = i + tl.arange(0, n)  # idx: [i, i+1, ..., n-1]
        mask = idx <= i
        x += tl.where(mask, -tl.sum(U[: i + 1, i] * x[idx[mask]], axis=0) / U[i, i], 0.0)
    x += tl.where(tl.arange(0, n) <= 0, y, 0.0)
    return x

@triton.jit
def fused_lu_solve(A, b):
    n = A.shape[0]
    # A = P @ L @ U
    P, L, U = householder.triangular_factorization(A)
    # L @ U @ x = b
    x = _lu_solve(L, U, b)
    return x

def fused_lu_solve(A: Tensor, b: Tensor) -> Tensor:
    """
    Args:
        A: The input matrix ``A`` of shape ``[..., n, n]``.
        b: The right-hand side tensor ``b`` of shape ``[..., n]``.

    Returns:
        Tensor: The solution ``x`` of shape ``[..., n]``.
    """
    A = A.contiguous()
    b = b.contiguous()
    if A.shape[-1] != b.shape[-1]:
        raise ValueError(f"Shape must be the same, but got {A.shape} and {b.shape}")
    if A.ndim < 2:
        raise ValueError(f"Input ndim must be larger than 1, but got {A.ndim}")

    n = A.shape[-1]
    if n == 1:
        return torch.tensor([[b / A]], dtype=A.dtype, device=A.device)
    elif n <= 128:
        return torch.linalg.solve(A, b)
    else:
        block_size = triton.next_power_of_2(n)
        if block_size > 1024:
            block_size = 512
        grid = (b.shape[:-2] + (1,))
        x = torch.empty(grid + (n,), dtype=A.dtype, device=A.device)
        LU = torch.empty(grid + (n, n), dtype=A.dtype, device=A.device)
        P = torch.empty(grid + (n, n), dtype=torch.int32, device=A.device)

        _fused_lu_solve(
            A, b, LU, P, x, n=n, block_size=block_size, grid=grid, dtype=A.dtype
        )
        return x

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
    ],
    key=["n"],
)
@triton.jit
def _fused_lu_solve(
    A, b, LU, P, x, n, block_size, grid, dtype: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Load A, b to SRAM
    A = A + pid * n * n
    b = b + pid * n
    if dtype == torch.float32:
        A = A.to(tl.float32)
        b = b.to(tl.float32)
    # A = P @ L @ U
    P, L, U = householder.triangular_factorization(A, n, block_size, 1)
    # Solve L @ U @ x = b
    _lu_solve(L, U, b, x, n, block_size)
