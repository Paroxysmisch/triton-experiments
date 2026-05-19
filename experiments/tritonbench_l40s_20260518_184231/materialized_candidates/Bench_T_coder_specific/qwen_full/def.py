import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume


def heur_n_warps(n, **meta):
    return min(triton.cdiv(128, n), 8)


def heur_eq_blocksize(meta):
    return meta["n"] <= 128


@triton.heuristics(
    {
        "n_warps": heur_n_warps,
        "eq_blocksize": heur_eq_blocksize,
    }
)
@triton.jit
def _lu_solve(
    A,
    L,
    U,
    P,
    b,
    x,
    n: tl.constexpr,
    batch: tl.constexpr,
    rhs: tl.constexpr,
    idx: tl.constexpr,
    n_warps: tl.constexpr,
    eq_blocksize: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if idx == 0:
        if not eq_blocksize:
            pid = pid * n + tl.arange(0, n)
        # Compute column block pointers
        Acol = A + pid * rhs
        Lcol = L + pid * n
        Ucol = U + pid * n
        b_ptr = b + pid * rhs
        x_ptr = x + pid * rhs
        # Compute P^T b
        if idx == 0:
            for i in range(n):
                mask = (i + 1) * rhs + tl.arange(0, rhs) < volume(b)
                p_idx = tl.load(P + i * rhs + tl.arange(0, rhs)).to(tl.int64)
                x_ptr += tl.where(
                    mask, tl.load(b_ptr + p_idx * rhs + tl.arange(0, rhs)), 0
                )
        for i in range(n):
            # Compute y = L^-1 (P^T b)
            mask = i * rhs + tl.arange(0, rhs) < volume(b)
            load = tl.load(Acol + i * rhs + tl.arange(0, rhs), mask=mask)
            load_l = tl.load(Lcol + i * rhs + tl.arange(0, rhs), mask=mask)
            if i > 0:
                load_l += tl.sum(
                    tl.load(Lcol + i * rhs + tl.arange(0, i) * rhs).to(tl.float32)
                    * tl.load(x_ptr + tl.arange(0, i) * rhs).to(tl.float32),
                    axis=0,
                ).to(tl.float32)
            x_ptr += tl.where(
                mask, x_ptr - load_l * (load.to(tl.float32)), 0
            ).to(tl.float32)
        for i in range(n - 1, -1, -1):
            # Compute x = U^-1 y
            mask = i * rhs + tl.arange(0, rhs) < volume(b)
            tl.store(
                x_ptr + i * rhs + tl.arange(0, rhs),
                tl.where(
                    mask,
                    (x_ptr + i * rhs + tl.arange(0, rhs))
                    / tl.load(Ucol + i * rhs + tl.arange(0, rhs)),
                    0,
                ),
            )


def _lu_solve_wrapper(A, b, P, L, U, *, pivot, out):
    n = L.shape[-1]
    batch = volume(L.shape[:-2])
    rhs = b.shape[-1]
    x = out if out is not None else torch.empty_like(b)
    grid_fn = lambda meta: (triton.cdiv(n, meta["block_size"]),)
    _lu_solve[grid_fn](
        A, L, U, P, b, x, n, batch, rhs, idx=0, eq_blocksize=heur_eq_blocksize(n)
    )
    return x
