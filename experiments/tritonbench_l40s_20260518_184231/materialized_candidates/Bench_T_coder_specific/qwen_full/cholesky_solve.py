import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_solve(
    B, L, upper, batch_shape, n, k, stride_bat, stride_b_m, stride_b_n, stride_l_m, stride_l_n, **meta
):
    # compute indices
    i_m = tl.arange(0, n)
    i_n = tl.arange(0, k)

    # get elements of B and L
    if upper:
        # B
        b_bat = tl.broadcast_to(tl.arange(0, batch_shape[0]).reshape((batch_shape[0], 1)), (batch_shape[0], k))
        b_idx = b_bat * stride_bat + i_n * stride_b_n + i_m[:, None] * stride_b_m
        b = tl.load(B + b_idx, mask=(i_n[None, :] < i_m[:, None]) & (i_m[:, None] < n), other=0.0)

        l_bat = tl.broadcast_to(tl.arange(0, batch_shape[0]).reshape((batch_shape[0], 1)), (batch_shape[0], n))
        l_idx = l_bat * stride_bat + i_m[:, None] * stride_l_m + i_n[None, :] * stride_l_n
        l = tl.load(L + l_idx, mask=(i_m[:, None] < n) & (i_n[None, :] < n), other=0.0)
        l = tl.conj(l)
    else:
        # B
        b_bat = tl.broadcast_to(tl.arange(0, batch_shape[0]).reshape((batch_shape[0], 1)), (batch_shape[0], k))
        b_idx = b_bat * stride_bat + i_m[:, None] * stride_b_m + i_n[None, :] * stride_b_n
        b = tl.load(B + b_idx, mask=(i_m[:, None] < i_n[None, :]) & (i_n[None, :] < n), other=0.0)

        l_bat = tl.broadcast_to(tl.arange(0, batch_shape[0]).reshape((batch_shape[0], 1)), (batch_shape[0], n))
        l_idx = l_bat * stride_bat + i_n[None, :] * stride_l_n + i_m[:, None] * stride_l_m
        l = tl.load(L + l_idx, mask=(i_n[None, :] < n) & (i_m[:, None] < n), other=0.0)

    # compute
    if meta["dtype"] in (tl.float32, tl.cfloat):
        b = b.float32()
        l = l.float32()
        c = tl.dot(b.to(l), tl.conj(l).to(l), allow_tf32=True)
    else:
        b = b.to(l.dtype)
        c = tl.dot(b, tl.conj(l).to(l), allow_tf32=True)

    # store
    c_bat = tl.broadcast_to(tl.arange(0, batch_shape[0]).reshape((batch_shape[0], 1)), (batch_shape[0], k))
    c_idx = c_bat * stride_bat + i_n[None, :] * stride_b_n + i_m[:, None] * stride_b_m
    tl.store(B + c_idx, c, mask=(i_n[None, :] < i_m[:, None]) & (i_m[:, None] < n))


def cholesky_solve(b, L, upper=False, out=None):
    b = b.contiguous()
    L = L.contiguous()

    if out is None:
        c = torch.empty_like(b)
    else:
        c = out

    batch_shape = b.shape[:-2]
    n = b.shape[-2]
    k = b.shape[-1]
    assert L.shape[-1] == L.shape[-2] == n, "L must be n-by-n"
    assert b.shape[-2:] == L.shape[-2:], "b and L must be broadcastable in their last two dimensions"

    if len(batch_shape) > 0:
        b = b.reshape(-1, n, k)
        L = L.reshape(-1, n, n)
        c = c.reshape(-1, n, k)
        n_batch = batch_shape[0]
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_N"]) * triton.cdiv(k, meta["BLOCK_SIZE_K"]), n_batch)
        _cholesky_solve[grid](b, L, upper, batch_shape, n, k, *b.stride(), *L.stride(), **b.dtype.meta)
        c = c.reshape(*batch_shape, n, k)
    else:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_N"]), triton.cdiv(k, meta["BLOCK_SIZE_K"]))
        _cholesky_solve[grid](b, L, upper, batch_shape, n, k, *b.stride(), *L.stride(), **b.dtype.meta)
    return c
