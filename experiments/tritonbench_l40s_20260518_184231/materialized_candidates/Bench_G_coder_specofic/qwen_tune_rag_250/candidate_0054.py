import torch
import triton
import triton.language as tl


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h0, ht, h,
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_g_h, s_g_t, s_g_d,
    s_h_h, s_h_t, s_h_d,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NH: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = tl.num_programs(1)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, K), (s_g_t, s_g_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, K, V), (s_h_t, s_h_d), (i_t * BT, i_k * BK, i_v * BV), (BT, BK, BV), (1, 0))
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_g2 = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BK)[None, :], b_g, 0.)
        b_h *= tl.math.exp2(-b_g2)
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
        if i_t == tl.cdiv(T, BT) - 1 and STORE_FINAL_STATE:
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    if not STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + (i_t * BT - 1) * s_h_t, (T, K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def fwd_inner(k, v, g, o, B, H, T, K, V, BT, BK, BV, USE_INITIAL_STATE, STORE_FINAL_STATE):
    NT = triton.cdiv(T, BT)
    NH = B * H
    grid = (triton.cdiv(K, BK), triton.cdiv(V, BV), NH) if NT > 1 else (triton.cdiv(V, BV), NH)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, o if USE_INITIAL_STATE else None, o if STORE_FINAL_STATE else None, v,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        g.stride(1), g.stride(2), g.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NH=NH,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        STORE_FINAL_STATE=STORE_FINAL_STATE
    )
