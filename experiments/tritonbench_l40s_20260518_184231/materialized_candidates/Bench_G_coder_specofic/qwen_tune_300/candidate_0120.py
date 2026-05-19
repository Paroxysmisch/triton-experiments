import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    b: tl.constexpr,
    h: tl.constexpr,
    t: tl.constexpr,
    bt: tl.constexpr,
    bf: tl.constexpr,
    scalar,
):
    i_f, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * t * b, (b, t), (t, 1), (i_f * bf, 0), (bf, bt), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * t * b, (b, t), (t, 1), (i_f * bf, 0), (bf, bt), (1, 0))
    mask = (i_f * bf + tl.arange(0, bf)) < b

    bs = tl.load(p_s, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if scalar:
        bs += i_f
    cs = tl.cumsum(bs, axis=1)
    cs = tl.where(mask, cs, 0)
    cs = cs + (tl.sum(bs, axis=1) - tl.sum(cs, axis=1))[:, None] * (tl.arange(0, bt) >= bt - 1)
    tl.store(p_o, cs.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_global_cumsum_scalar(s, dtype=None):
    if dtype is None:
        dtype = s.dtype
    b, h, t = s.shape
    z = torch.empty_like(s, dtype=dtype)
    bt = 32
    bf = 64
    chunk_global_cumsum_scalar_kernel[(b // bf, h)](s, z, b, h, t, bt, bf, False)
    return z


@triton.jit
def chunk_global_cumsum_scalar_kernel_i(
    s,
    o,
    b: tl.constexpr,
    h: tl.constexpr,
    t: tl.constexpr,
    bt: tl.constexpr,
    bf: tl.constexpr,
    i,
):
    i_f, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * t * b, (b, t), (t, 1), (i_f * bf, 0), (bf, bt), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * t * b, (b, t), (t, 1), (i_f * bf, 0), (bf, bt), (1, 0))
    mask = (i_f * bf + tl.arange(0, bf)) < b

    bs = tl.load(p_s, boundary_check=(0, 1), padding_option="zero")
    if i == -1:
        bs += i_f
    cs = tl.cumsum(bs, axis=1)
    cs = tl.where(mask, cs, 0)
    cs = cs + (tl.sum(bs, axis=1) - tl.sum(cs, axis=1))[:, None] * (tl.arange(0, bt) >= bt - 1)
    tl.store(p_o, cs.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_global_cumsum_scalar_i(s, i, dtype=None):
    if dtype is None:
        dtype = s.dtype
    b, h, t = s.shape
    z = torch.empty_like(s, dtype=dtype)
    bt = 32
    bf = 64
    chunk_global_cumsum_scalar_kernel_i[(b // bf, h)](s, z, b, h, t, bt, bf, i)
    return z


def chunk_global_cumsum_scalar_int(s):
    b, h, t = s.shape
    z = torch.empty(b, h, t, dtype=torch.int64, device=s.device)
    for i in range(-1, 100):
        x = chunk_global_cumsum_scalar_i(s, i)
        z = z + (x - s) * (i + 1)
    return z
