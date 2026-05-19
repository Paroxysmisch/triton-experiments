import torch
import triton
import triton.language as tl


@triton.jit
def parallel_rebased_fwd_kernel(
    q, k, v, o, z,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_SCALE: tl.constexpr, NORMALIZE: tl.constexpr
):
    # Triton kernel code for forward pass
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, BK), (s_qk_t, s_qk_d), (i_k * BK, 0), (BT, BK), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (BK, T), (s_qk_d, s_qk_t), (0, i_k * BK), (BK, BT), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, BV), (s_vo_t, s_vo_d), (i_k * BV, 0), (BT, BV), (0, 1))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, BV), (s_vo_t, s_vo_d), (i_v * BV, 0), (BT, BV), (0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    if USE_SCALE:
        b_q = b_q * scale
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for s in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_SCALE:
            b_k = b_k * scale
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BT))
    tl.debug_barrier()
    b_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_s, 0)
    if NORMALIZE:
        b_z = tl.max(b_s, axis=1, keepdims=True)
        b_s = b_s - b_z
        b_z = tl.math.log(tl.sum(tl.exp(b_z - b_s))) + b_z
        b_s = b_z + b_s - b_z
    else:
        b_z = tl.max(b_s, axis=1, keepdims=True)
    b_s = tl.math.exp(b_s - b_z)
    tl.store(z + i_bh * T + i_k * BT + tl.arange(0, BT), b_z)
    b_o = tl.dot(b_s.to(b_q.dtype), tl.trans(b_v), allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty))


@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh, i_k,
    q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_SCALE: tl.constexpr
):
    # Triton kernel code for backward pass
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, BV), (s_vo_t, s_vo_d), (i_k * BV, 0), (BT, BV), (0, 1))
    p_dz = tl.make_block_ptr(dz + i_bh * s_vo_h, (T, BV), (s_vo_t, s_vo_d), (i_k * BV, 0), (BT, BV), (0, 1))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, BK), (s_qk_t, s_qk_d), (i_k * BK, 0), (BT, BK), (0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dz = tl.load(p_dz, boundary_check=(0, 1))
    b_do = b_do.to(k.dtype.element_ty)
    b_dz = b_dz.to(k.dtype.element_ty)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (BK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, BV), (s_vo_t, s_vo_d), (i_k * BV, 0), (BT, BV), (0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    if USE_SCALE:
        b_k = b_k * scale
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dq = tl.dot(b_do.to(b_k.dtype), tl.trans(b_v), allow_tf32=False) + tl.dot(b_dz.to(b_k.dtype), tl.trans(b_v), allow_tf32=False)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty))


@triton.jit
def _parallel_rebased_bwd_dkv(
    i_bh, i_k,
    q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_SCALE: tl.constexpr
):
    # Triton kernel code for backward pass
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, BK), (s_qk_t, s_qk_d), (i_k * BK, 0), (BT, BK), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, BV), (s_vo_t
