import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, scale,
    h0,
    s_k, s_v, s_h,
    B, H, T,
    BT: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_bh = tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)

    b_o, b_h = tl.zeros([DK, DV], dtype=tl.float32), tl.zeros([DK, DV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * DK * DV, (DK, DV), (DV, 1), (0, i_v * DV), (DK, DV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + (i_bh + i_v * B * H) * s_k, (DK, T), (1, s_k), (0, i * BT), (DK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v, (T, DV), (s_v, 1), (i * BT, i_v * DV), (BT, DV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_k, tl.trans(b_k), allow_tf32=False)
        b_s = (b_s * scale).to(b_k.dtype)
        b_h = b_s * b_h + tl.dot(b_k, tl.trans(b_v.to(b_k.dtype)), allow_tf32=False)
        b_o = b_o * tl.math.exp2(tl.amax(b_s, axis=1))[:, None] + tl.dot(b_k, tl.trans(b_h), allow_tf32=False)

        if STORE_FINAL_STATE:
            p_ho = tl.make_block_ptr(h0 + (i_bh + i_v*B*H) * DK * DV, (DK, DV), (DV, 1), (i_h * DK * DV, i_v * DV), (DK, DV), (1, 0))
            tl.store(p_ho, tl.trans(b_o).to(p_ho.dtype.element_ty), boundary_check=(0, 1))

        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))


@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_stages=1, num_warps=2),
        triton.Config({'BT': 16}, num_stages=2, num_warps=4),
        triton.Config({'BT': 16}, num_stages=3, num_warps=8),
        triton.Config({'BT': 32}, num_stages=1, num_warps=2),
        triton.Config({'BT': 32}, num_stages=2, num_warps=4),
        triton.Config({'BT': 32}, num_stages=3, num_warps=8),
        triton.Config({'BT': 64}, num_stages=1, num_warps=2),
        triton.Config({'BT': 64}, num_stages=2, num_warps=4),
        triton.Config({'BT': 64}, num_stages=3, num_warps=8)
    ],
    key=['DV', 'DK']
)
@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h,
    s_q, s_k, s_v, s_h,
    o,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    NUM_STAGES: tl.constexpr
):
    i_v, i_bh = tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)

    i_h = i_bh % H
    b_bh = tl.math.log2(1 - tl.math.pow(2, -5 - i_h*1.))
    d_b, d_o, d_h = tl.math.exp2(BT * b_bh), tl.math.exp2((o_i+1) * b_bh), tl.math.exp2((BT - o_i - 1) * b_bh)
    decay_o, decay_h = d_o, d_b

    m_s = o_i[:, None] > o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_bh), 0)

    p_q = tl.make_block_ptr(q + i_bh * s_q, (T, DK), (s_q, 1), (0, 0), (BT, DK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k, (T, DK), (s_k, 1), (0, 0), (BT, DK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v, (T, DV), (s_v, 1), (0, i_v * BV), (BT, DV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h, (T, DV), (s_h, 1), (0, i_v * BV), (BT, DV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh+i_v*B*H) * s_q, (T, DK), (s_q, 1), (0, 0), (BT, DK), (1, 0))

    for i in range(0, tl.cdiv(T, BT)):
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o = tl.dot(b_
