import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_TR': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_TR': 128}, num_warps=8),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h,
    g,
    do, dh,
    dq, dk, dg,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_DH: tl.constexpr,
    BLOCK_SIZE_TR: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    i_b = i_bh // H

    o_qk = i_b * s_qk_h + i_h * BT
    o_v = i_b * s_vo_h + i_h * BT

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32) if USE_DH else None

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BK, BT], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    b_dg = tl.zeros([BT, BV], dtype=tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + o_qk, (T - i * BT, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + o_qk, (DK, T - i * BT), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_h * BT, (T - i * BT, DK), (s_vo_t, s_vo_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + o_qk, (T - i * BT, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_do = tl.make_block_ptr(do + o_qk, (T - i * BT, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + o_qk, (T - i * BT, DV), (s_qk_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + o_v, (T - i * BT, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(b_dq + i * BT * BK, (T - i * BT, DV), (BK, BV), (0, i_v * BV), (BK, BV), (1, 0))
        p_dk = tl.make_block_ptr(b_dk + i_k * BK * BT, (BK, T - i * BT), (BV, BT), (i_v * BV, i * BT), (BV, BT), (0, 1))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty))
        if USE_DH:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_g = tl.load(p_dg, boundary_check=(0, 1)).to(b_do.dtype)
        b_h += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        if USE_DH:
            b_dh += tl.dot(b_q, b_dh.to(b_q.dtype), allow_tf32=False)
        b_g = tl.exp(b_g * scale)
        b_dv += tl.dot(b_do * b_g, b_v, allow_tf32=False)
        b_dq += tl.dot(b_do * b_g, b_h.to(b_do.dtype), allow_tf32=False)
        b_dk += tl.dot(b_v, b_dv.to(b_v.dtype), allow_tf32=False)
        b_dk = -tl.dot(b_k, b_dk.to(b_k.dtype), allow_tf32=False) + b_dk

        tl.store(p_h, b_h.to(p_h.dtype.element_ty))
        if USE_DH:
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty))

    o_dq = i_b * s_qk_h + i_k * BK * BT
    o_dk = i_b * s_qk_h + i_k * BK * BT
    o_dg = i_b * s_vo_h + i_v * BV * BT

    o_bq = i_k * BT + tl.arange(0, BLOCK_SIZE_TR)
    o_bk = i_k * BT + tl.arange(0, BLOCK_SIZE_TR)
    o_bb = tl.arange(0, BLOCK_SIZE_TR)

    p_dq = tl.make_block_ptr(dq + o_dq, (DK, B * H * BT), (s_qk_d, s_qk_t, s_qk_d), (i_v * BV, i_k * BT, i_h * BT), (DV, 1, 1), (0, 1, 0))
    p_dk = tl.make_block_ptr(dk + o_dk, (B * H * BT, DK), (s_qk_d, s_qk_t, s_qk_d), (i_k * BT, i_v * BV, i_h * BT), (1, DV, 1), (0, 1, 0))
    p_dg = tl.make_block_ptr(dg + o_dg, (DV, B * H * BT), (s_vo_d, s_vo_t, s_vo_d), (i_v * BV, i_k * BT, i_h * BT), (1, DV, 1), (0, 1, 0))

    p_bq = tl.make_block_ptr(b_dq, (BT, BK), (BK, 1), (o_bq, i_k * BK), (1, 0))
    p_bb = tl.make_block_ptr(b_dk, (BK, BT), (1, BT), (i_k * BK, o_bk), (0, 1))
    p_bg = tl.make_block_ptr(b_dg, (BT, BV), (BV, 1), (o_bk, i_v * BV), (1, 0))

    tl.store(p_dq, tl.load(p_bq).to(p_dq.dtype.element_ty).to(p_dq.dtype.element_ty))
    tl.store(p_dk, tl.load(p_bb).to(p_dk.dtype.element_ty).to(p_dk.dtype.element_ty))
    tl.store(p_dg, tl.load(p_bg).to(p_dg.dtype.element_ty).to(p_dg.dtype.element_ty))


def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh):
    B, H, T, DK, DV = *q.shape, v.shape[-1]
    scale = DK ** -0.5

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)

    def grid(BT, BK, BV):
        return (triton.cdiv(DV, BV) * triton.cdiv(DK, BK), triton.cdiv(DK, BK), B * H)

    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h,
        g,
        do, dh,
        dq, dk, dg,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        B, H, T, scale,
        DK=DK, DV=DV, USE_DH=(dh is not None), BT=32, BK=triton.next_power_of_2(DK), BV=triton.next_power_of_2(DV),
        BLOCK_SIZE_TR=32,
