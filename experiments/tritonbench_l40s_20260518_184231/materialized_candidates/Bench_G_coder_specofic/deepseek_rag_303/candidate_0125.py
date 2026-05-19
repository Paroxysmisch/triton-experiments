import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q,  # [B, H, L, K]
    k,  # [B, H, L, K]
    v,  # [B, H, L, V]
    o,  # [B, H, L, V]
    z,  # [B, H, L]
    s_qk_h,  # stride size: L * K
    s_qk_t,  # stride size: K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * V
    s_vo_t,  # stride size: V
    s_vo_d,  # stride size: 1
    scale,  # K ** -0.5, precalculated
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # H
    T: tl.constexpr,  # T
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_SCALE: tl.constexpr,  # whether to scale Q
    USE_NORMALIZE: tl.constexpr,  # whether to normalize attention logits
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_k = tl.arange(0, BTS) + i_c * BTS
    d_h = tl.math.exp2((BTS - o_k) * b_b)

    p_q = tl.make_block_ptr(
        q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_kv * BK), (BTL, BK), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_kv * BK), (BTS, BK), (0, 1)
    )
    p_v = tl.make_block_ptr(
        v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_kv * BV), (BTL, BV), (1, 0)
    )

    p_o = tl.make_block_ptr(
        o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), ((i_c+1)*BTL - 1, i_kv * BV), (BTL, BV), (-1, 0)
    )
    p_z = tl.make_block_ptr(
        z + i_bh * s_vo_h, (T,), (s_vo_t,), ((i_c + 1) * BTL - 1,), (BTL,), (-1,)
    )

    # init accumulator
    b_s = tl.zeros([BTL, BV], dtype=tl.float32)

    if USE_SCALE:
        b_q = tl.load(p_q, boundary_check=(0, 1))
        q_i = (b_q * scale).to(b_q.dtype)
    else:
        q_i = tl.load(p_q, boundary_check=(0, 1))

    tl.store(p_o, q_i.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    for _ in range((i_c + 1) * BTL, T, BTS):
        # update pointers
        p_q = tl.advance(p_q, (BTL, 0))
        p_o = tl.advance(p_o, (-1, 0))
        p_z = tl.advance(p_z, (-1,))

        o_k += BTS
        d_h = d_h[:-1] * tl.math.exp2(-b_b * BTS)
        d_h = tl.concatenate([d_h, tl.zeros([1], dtype=d_h.dtype)], axis=0)

        if USE_SCALE:
            b_q = tl.load(p_q, boundary_check=(0, 1))
            q_i = (b_q * scale).to(b_q.dtype)
        else:
            q_i = tl.load(p_q, boundary_check=(0, 1))
        # [BTS, BK]
        if i_kv % (TLV // V) == 0:
            # [BTS, BV]. And, skip bank conflict
            p_v = tl.advance(p_v, (-BTS, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BTS, BV]
        # caution, bank conflict may happen, need to transpose first, see above
        b_q = tl.trans(q_i)  # [BTS, BK]

        # [BTL, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BTL, BTS]
        # initialize b_s to 0 and keep adding on a running sum
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

        if USE_NORMALIZE:
            # similar to chunk_size
            b_s *= d_h[:, None]
            # normal case
        # end if
        # [BTL, BV]
        b_v = b_v * d_h[:, None]

        b_o = b_s + b_v  # [BTL, BV]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_z, tl.sum(b_o, axis=0), boundary_check=0)
        # set zero
        b_s = tl.zeros([BTL, BV], dtype=tl.float32)
    return

@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_kv,
    i_h,
    k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t,
    s_vo_d, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BTL: tl.constexpr, BTS: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr
):
    # key and value first, so as to save registers
    # dq first
    # scale
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h
