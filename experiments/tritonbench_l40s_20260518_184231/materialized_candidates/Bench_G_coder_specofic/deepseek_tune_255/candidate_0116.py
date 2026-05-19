import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q,
    k,
 v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dg,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    T,
    K,
    V,
    scale,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    b_dq = tl.zeros([BT, BK, BV], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK, BV], dtype=tl.float32)
    b_dg = tl.zeros([BT, BT, BK], dtype=tl.float32)

    i_k = tl.arange(0, BT)
    i_v = tl.arange(0, BV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (BT, BK), (s_qk_t, s_qk_d), (i_k, i_v), (BT, BV), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (BK, BT), (s_qk_d, s_qk_t), (i_v, i_k), (BV, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (BT, BV), (s_vo_t, s_vo_d), (i_k, i_v), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_vo_h, (BT, BV), (s_vo_t, s_vo_d), (i_k, i_v), (BT, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (BT, BV), (s_vo_t, s_vo_d), (i_k, i_v), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * s_vo_h, (BT, BV), (s_vo_t, s_vo_d), (i_k, i_v), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (BT, BT), (s_qk_t, s_qk_d), (i_k, i_k), (BT, BT), (1, 0))

    mask = i_k[:, None] >= i_k[None, :]

    for i in range(V):
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_dh = tl.load(p_dh, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)

        b_dq += b_dh * b_g[:, None] * scale
        b_dk += tl.trans(b_q)[None, :, :] * (b_do[:, None, :] * b_g[None, :, :] * scale)

        b_dh = tl.dot(b_g.to(tl.float32), b_do, allow_tf32=False)
        b_do = b_h * tl.trans(b_k)[:, None, :]
        b_g = b_g * tl.trans(b_k)[:, None] * b_k[None, :, None]

        p_q = tl.advance(p_q, (1, 0))
        p_k = tl.advance(p_k, (0, 1))
        p_v = tl.advance(p_v, (1, 0))
        p_h = tl.advance(p_h, (1, 0))
        p_do = tl.advance(p_do, (1, 0))
        p_dh = tl.advance(p_dh, (1, 0))
        p_g = tl.advance(p_g, (1, 1))

    tl.store(b_dq.to(dq.dtype.element_ty), dq + i_bh * s_qk_h, mask=(i_k[:, None], i_v[None, :]))
    tl.store(b_dk.to(dk.dtype.element_ty), dk + i_bh * s_qk_h, mask=(i_k[:, None], i_v[None, :]))
    tl.store(b_dg.to(dg.dtype.element_ty), dg + i_bh * s_qk_h, mask=(i_k[:, None], i_k[None, :]))

def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh, scale):
    B, H, T, K, V = *q.shape, v.shape[-1]
    BT = 128
    BK = 128
    BV = 64

    grid = (K, H * B, 1)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q,
        k,
        v,
        h,
        g,
        do,
        dh,
        q,
        k,
        g,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        T,
        K,
        V,
        scale,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return q, k, g
