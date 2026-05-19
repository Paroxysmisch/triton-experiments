import torch
import triton
import triton.language as tl

def _lu(A, pivot=True):
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    A = A.reshape(-1, n, n)
    if pivot:
        p, l, u = triton.ops.pivoted_lu(A)
    else:
        l, u = triton.ops.lu(A)
    return p.reshape(*batch_shape, n) if pivot else l.reshape(*batch_shape, n), u.reshape(*batch_shape, n)


def _lu_solve(lu, _, B, transpose: bool = False, min_triangular_size: int = 16, pivot=True):
    batch_shape = lu.shape[:-2]
    n = lu.shape[-1]
    k = B.shape[-1]
    lu = lu.reshape(-1, n, n)
    B = B.reshape(-1, n, k)
    if transpose:
        l, u = lu.transpose(-2, -1).unbind(-1)
    else:
        l, u = lu.unbind(-1)
    if pivot:
        p, l, u = triton.ops.pivoted_lu(lu)
        B = torch.matmul(p.unsqueeze(-1), B)
    if min_triangular_size < n:
        l_lo, l_hi = l.split([1, n - 1], dim=-1)
        u_lo, u_hi = u.split([n - 1, 1], dim=-2)
        B_lo, B_hi = B.split([1, k], dim=-1)
        l_lo = l_lo.squeeze(-1)
        u_hi = u_hi.squeeze(-2)
        B_lo = B_lo.squeeze(-1)
        x_lo = torch.tril_solve(B_lo, l_lo, runumerics=False)
        x_hi = torch.triu_solve(B_hi, u_hi, runumerics=False) + torch.einsum('...ij,...j->...i', l_hi, x_lo)
        x = torch.cat([x_lo.unsqueeze(-2), x_hi.unsqueeze(-3)], dim=-2)
    else:
        B = B.squeeze(-1)
        x = torch.tril_solve(B, l, runumerics=False)
    return x.reshape(*batch_shape, k) if transpose else x.reshape(*batch_shape, k)
