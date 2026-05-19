import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def parallel_rebased_fwd_kernel(
    q, k, v, o, z,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_SCALE: tl.constexpr,
    USE_NORMALIZE: tl.constexpr,
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    i_b = i_bh // H

    p_q = tl.make_block_ptr(q + i_b * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_b * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_b * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype) if USE_SCALE else b_q
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    if USE_NORMALIZE:
        b_z = tl.zeros([BTL], dtype=tl.float32) - float(T)
    else:
        b_z = tl.zeros([BTL], dtype=tl.float32) + 1.0 / float(DV)

    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_b * s_qk_h, (DK, T),
                            (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_b * s_vo_h, (T, DV),
                            (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        if USE_NORMALIZE:
            b_s = b_s * b_s
            b_z += tl.sum(tl.where(m_s, b_s, 0), axis=1)
        b_o += tl.dot(tl.where(m_s, b_s, 0).to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh * T + i_c * BTL) * DV, (T, DV),
                            (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh * T + i_c * BTL) + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if USE_NORMALIZE:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty),
                 mask=((i_c * BTL + tl.arange(0, BTL)) < T)))
    else:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty))


@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh, i_c, i_k, i_v, i_h,
    q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
    s_vo_t, s_vo_d, B, H, T, scale,
    BTL: tl.constexpr, BTS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, USE_SCALE: tl.constexpr,
    USE_NORMALIZE: tl.constexpr, BLOCK_PTR: tl.constexpr
):
    p_do = tl.make_block_ptr(do + i_b * s_vo_h, (T, DV), (s_vo_t, s_vo_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_b * s_qk_h, (T, DK),
                            (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    if BLOCK_PTR:
        p_k = tl.make_block_ptr(k + i_b * s_qk_h, (T, DK),
                                (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_b * s_vo_h, (DV, T),
                                (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    else:
        p_k = tl.make_block_ptr(k + i_b * s_qk_h, (DK, T),
                                (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
        p_v = tl.make_block_ptr(v + i_b * s_vo_h, (T, DV),
                                (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = (b_do * scale).to(b_do.dtype) if USE_SCALE else b_do
    if USE_NORMALIZE:
        b_dz = tl.load(dz + i_b * T + i_c * BTL) * 2.0
        b_z = tl.load(dz + i_b * T + i_c * BTL + tl.arange(0, BTL))
    else:
        b_dz = tl.zeros([BTL], dtype=tl.float32) + 2.0 / float(DV)
        b_z = tl.zeros([BTL], dtype=tl.float32) + 1.0 / float(DV)

    for _ in range(0, i_c * BTL, BTS):
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if USE_NORMALIZE:
            b_ds = b_ds * b_ds
            b_dz += tl.sum(b_ds, axis=1)
        b_dq = b_dq + tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        b_do = b_do - \
            tl.dot(b_q * b_z[:, None], b_v, allow_tf32=False)
        b_do = (b_do * scale).to(b_do.dtype) if USE_SCALE else b_do
        p_k = tl.advance(p_k, (
