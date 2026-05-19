import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor.triton_heuristics import whitelist
from torch._inductor.utils import instance_descriptor

@whitelist(
    {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"},
    {"full_matrices", "rcond", "out"},
)
@triton.jit
def _pseudoinverse_svd(
    A,
    full_matrices,
    rcond,
    out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = tl.num_programs(0)

    if not full_matrices:
        n = tl.num_programs(1)
        k = BLOCK_SIZE_N
        offs_am = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = A + offs_am[:, None] * n + offs_bn[None, :]
        a_re = tl.load(a_ptrs, mask=offs_bn[None, :] < k, other=0.0)
        a_im = tl.load(a_ptrs, mask=offs_bn[None, :] >= k, other=0.0)
        a_re = a_re.to(tl.float32)
        a_im = a_im.to(tl.float32)
        a_shape = (m, k)
    else:
        k = tl.num_programs(1)
        n = BLOCK_SIZE_N
        offs_am = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = A + offs_am[:, None] * k + offs_bn[None, :]
        a_re = tl.load(a_ptrs, mask=offs_am[:, None] < m, other=0.0)
        a_im = tl.load(a_ptrs, mask=offs_am[:, None] >= m, other=0.0)
        a_re = a_re.to(tl.float32)
        a_im = a_im.to(tl.float32)
        a_shape = (m, n)

    s_re, s_im, u_re, u_im, v_re, v_im = tl.linalg.svd(
        a_re, a_im, full_matrices=full_matrices
    )

    max_s = tl.max(s_re.to(tl.float32), axis=1)
    max_s = tl.max(max_s, axis=0)
    tol = rcond * max_s

    s_re = tl.where(s_re > tol, 1 / s_re, 0.0)
    s_im = tl.where(s_im > tol, -1 / s_im, 0.0)

    if not full_matrices:
        u_shape = (m, k)
        u = tl.trans(u_re) + 1j * tl.trans(u_im)
        v_shape = (k, n)
        v = v_re + 1j * v_im
    else:
        u_shape = (m, m)
        u = tl.trans(u_re) + 1j * tl.trans(u_im)
        v_shape = (n, n)
        v = v_re + 1j * v_im

    uv_shape = u_shape[1] + v_shape[1]
    u = tl.reshape(u, (u_shape[0], u_shape[1] * 2))
    v = tl.reshape(v, (v_shape[0], v_shape[1] * 2))

    uv_re = tl.zeros((u_shape[0], uv_shape), dtype=tl.float32)
    uv_im = tl.zeros((u_shape[0], uv_shape), dtype=tl.float32)

    offs_uv = tl.arange(0, uv_shape)
    uv_ptrs = u + offs_uv
    uv_re = tl.load(uv_ptrs, mask=offs_uv[None, :] < u_shape[1] * 2, other=0.0)
    uv_im = tl.load(uv_ptrs, mask=offs_uv[None, :] >= u_shape[1] * 2, other=0.0)

    v_ptrs = v + offs_uv
    uv_re += tl.trans(tl.load(v_ptrs, mask=offs_uv[None, :] < v_shape[1] * 2, other=0.0))
    uv_im += -tl.trans(
        tl.load(v_ptrs, mask=offs_uv[None, :] >= v_shape[1] * 2, other=0.0)
    )

    s = tl.reshape(s_re, (a_shape[0], a_shape[1] * 2))
    uv_re = uv_re.to(tl.float32)
    uv_im = uv_im.to(tl.float32)
    s = s.to(tl.float32)
    out_ptrs = out + offs_am[:, None] * uv_shape + offs_uv[None, :]
    tl.store(out_ptrs, uv_re)
    tl.store(out_ptrs, uv_im)


def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None) -> Tensor:
    if A.is_complex():
        dtype = A.dtype
    else:
        dtype = torch.float32

    if out is None:
        out = torch.empty(A.shape, dtype=dtype, device=A.device)

    if A.ndim - 2 > out.ndim:
        raise ValueError(
            f"Expected `out` to have a batch dimension with A, but found out.shape = {out.shape} and A.shape = {A.shape}"
        )

    batch_dims = A.shape[: -out.ndim - 2]
    in_shape = A.shape[-2:]
    out_shape = out.shape[-2:]

    if full_matrices:
        grid = lambda META: (
            triton.cdiv(in_shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(in_shape[1], META["BLOCK_SIZE_N"]),
        )
    else:
        grid = lambda META: (
            triton.cdiv(in_shape[0], META["BLOCK_SIZE_M"]),
            triton.cdiv(in_shape[1], META["BLOCK_SIZE_N"]),
        )

    _pseudoinverse_svd[grid](
        A,
        full_matrices,
        rcond,
        out,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )

    if A.ndim - 2 > out.ndim:
        out = out.sum(0, keepdim=True)

    out = out.reshape(batch_dims + out_shape)

    return out
