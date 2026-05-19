import torch
import triton
import triton.language as tl

@triton.jit
def _lu_no_pivoting_fwd_triton(
    A,
    L,
    U,
    batch,
    m,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    i_block = pid * BLOCK_SIZE
    i_row = i_block + tl.arange(0, BLOCK_SIZE)
    A = A + (i_row[:, None] * m + tl.arange(0, BLOCK_SIZE)[None, :])
    L = L + (i_row[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :])
    U = U + (i_row[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :])
    mask_l = i_row[:, None] >= tl.arange(0, BLOCK_SIZE)[None, :]
    mask_u = i_row[:, None] <= tl.arange(0, BLOCK_SIZE)[None, :]
    a = tl.load(A, mask=(i_row[:, None] < m) & (tl.arange(0, BLOCK_SIZE)[None, :] < n), other=0.0)
    zero = tl.zeros((1, 1), dtype=a.dtype)
    for k in range(BLOCK_SIZE):
        l = tl.where(mask_l, tl.sum(a / a[k, k] * tl.where(mask_u, U[k, k], zero)), zero)
        tl.store(L, l, mask=mask_l)
        u = tl.where(mask_u, a - tl.dot(L, U, allow_tf32=False), zero)
        tl.store(U, u, mask=mask_u)

def _lu_no_pivoting_fwd(A):
    batch, m, n = A.shape
    L = torch.empty(batch, m, m, dtype=A.dtype, device=A.device)
    U = torch.empty(batch, m, n, dtype=A.dtype, device=A.device)
    if m <= 128:
        BLOCK_SIZE = m
    else:
        BLOCK_SIZE = 128
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    _lu_no_pivoting_fwd_triton[grid](
        A,
        L,
        U,
        batch,
        m,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return L, U

def lu(A, *, pivot=True, out=None):
    if pivot:
        return torch.lu(A)
    else:
        assert A.is_cuda, "Input must be a CUDA tensor"
        if A.ndim == 2:
            return _lu_no_pivoting_fwd(A)
        else:
            batch, m, n = A.shape
            A = A.view(batch * m, n)
            L, U = _lu_no_pivoting_fwd(A)
            L = L.view(batch, m, m)
            U = U.view(batch, m, n)
            return L, torch.empty(0), U
