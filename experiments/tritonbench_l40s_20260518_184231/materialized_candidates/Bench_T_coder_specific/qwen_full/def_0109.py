import torch
import triton
import triton.language as tl

@triton.jit
def _pinv_svd_dag_cuda(
    U, S, V, rcond, m, n, bdim, full_matrices, swap_uv: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if bdim > 1:
        pid += tl.program_id(1) * tl.num_programs(0)
        bdim = 1

    col_offset = pid * BLOCK
    cols = tl.arange(0, BLOCK)
    mask = cols < n

    if full_matrices:
        row_offset = tl.arange(0, BLOCK)
        rows = tl.arange(0, BLOCK)
        full_mask = rows[:, None] < m
        mask = mask[None, :] & full_mask
    else:
        row_offset = col_offset

    u = tl.load(U + col_offset * m + row_offset, mask, other=0.0).to(tl.float32)
    if not swap_uv:
        v = tl.load(V + col_offset * m + row_offset, mask, other=0.0).to(tl.float32)
    else:
        v = tl.load(V + row_offset * n + col_offset, mask, other=0.0).to(tl.float32)

    s = tl.load(S + col_offset + tl.arange(0, BLOCK), mask=cols < n, other=0.0).to(
        tl.float32
    )
    s = tl.where(tl.abs(s) > rcond, 1.0 / s, 0.0)

    s = s.to(v.dtype)
    v = v.to(tl.float32)
    u = u.to(tl.float32)

    w = u * s
    if swap_uv:
        w = tl.dot(w, v, allow_tf32=False).to(v.dtype)
    else:
        w = tl.dot(w, v, allow_tf32=False).to(u.dtype)

    off = (col_offset + tl.arange(0, BLOCK)[None, :]) * m + (row_offset + tl.arange(0, BLOCK)[:, None])
    mask = (col_offset + tl.arange(0, BLOCK)[None, :]) < n
    mask &= (row_offset + tl.arange(0, BLOCK)[:, None]) < m

    tl.store(U + off, w, mask=mask)

def _pinv_svd_cuda(a, rcond, full_matrices, out):
    a = a.contiguous()
    if a.ndim == 2:
        a = a.unsqueeze(0)
    bdim = a.ndim - 2
    a = a.view(a.shape[:bdim] + (a.shape[-2] * a.shape[-1],))
    m, n = a.shape[-2:]
    s = min(m, n)
    out = torch.empty(a.shape[-2:] if out is None else out.shape, device=a.device, dtype=a.dtype)
    full_matrices = full_matrices and m >= n

    if a.dtype in (torch.cfloat, torch.cdouble):
        a = torch.linalg.cholesky(a).to(torch.float32)
        a = torch.triangular_solve(a, torch.conj(a).transpose(-2, -1), upper=True)[0]
    else:
        a, _, _ = torch.svd(a, full_matrices=False)

    if a.dtype in (torch.float32, torch.float64):
        s = torch.full((s,), 1.0, device=a.device, dtype=a.dtype)
        s = torch.index_fill_(s, 0, 0, 0)
    else:
        s = torch.full((s,), 1.0 + 0j, device=a.device, dtype=a.dtype)
        s = torch.index_fill_(s, 0, 0 + 0j, 0 + 0j)

    rcond = rcond * s[0]
    s = torch.where(tl.abs(s) > rcond, 1.0 / s, 0.0)

    if a.dtype in (torch.float32, torch.float64):
        out = torch.diag_embed(s)
    else:
        out = torch.diag_embed(s.to(torch.complex64))
    return out.view(a.shape[:bdim] + out.shape[-2:])

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> Tensor:
    if A.is_floating_point():
        if A.is_cuda:
            return _pinv_svd_cuda(A, rcond, full_matrices, out)
        elif A.is_cpu:
            raise NotImplementedError("SVD pseudoinverse is not supported on CPU.")
        else:
            raise RuntimeError("Invalid device for tensor.")
    else:
        raise RuntimeError("Pseudoinverse is only defined for floating point tensors.")
