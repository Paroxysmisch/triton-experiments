import torch
import triton
import triton.language as tl


@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s, o, s_s_h, s_s_t, s_s_d,
    T: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fwd_pre(g, B, H, T, S, BT):
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g,
        g.stride(1), g.stride(2), g.stride(3),
        T=T, S=S, BT=BT
    )
    return g


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h0, ht, h,
    s_k_h, s_k_t, s_k_d, s_k_m, s_v_h, s_v_t, s_v_d, s_v_m, s_g_h, s_g_t, s_g_d, s_g_m,
    H, T, K, V, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_v, i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    i_v, i_k = i_v * BV + tl.arange(0, BV), i_k * BK + tl.arange(0, BK)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if h0 is not None:
        p_h = tl.make_block_ptr(h0 + i_bh * s_k_h, (H, T, K), (s_k_t, s_k_d, s_k_m), (0, i_t - 1, i_k), (H, T, K), (0, 1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 2)).to(tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K, V), (s_k_t, s_k_d, s_k_m), (i_t * BT + tl.arange(0, BT), i_k, i_v), (BT, BK, BV), (1, 0, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, K, V), (s_v_t, s_v_d, s_v_m), (i_t * BT + tl.arange(0, BT), i_k, i_v), (BT, BK, BV), (1, 0, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, K, V), (s_g_t, s_g_d, s_g_m), (i_t * BT + tl.arange(0, BT), i_k, i_v), (BT, BK, BV), (1, 0, 0))
    for i in range(0, tl.cdiv(BT, BK)):
        b_k = tl.load(p_k, boundary_check=(0, 1, 2))
        b_v = tl.load(p_v, boundary_check=(0, 1, 2))
        b_g = tl.load(p_g, boundary_check=(0, 1, 2))
        b_h = tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False) * b_g + b_h
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 2))
        p_k = tl.advance(p_k, (BK, 0, 0))
        p_v = tl.advance(p_v, (BK, 0, 0))
        p_g = tl.advance(p_g, (BK, 0, 0))
    p_h = tl.make_block_ptr(ht + i_bh * s_k_h, (H, T, K), (s_k_t, s_k_d, s_k_m), (i_t * BT + tl.arange(0, BT), i_k), (BT, BK, BV), (1, 0, 0))
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 2))


def fwd_inner(k, v, g, h0, ht, h, B, H, T, K, V, BT, BK, BV, has_h0, no_reset, wire_weight, not_allow_tf32, extra_pass):
    BK, BV = min(BK, V), min(BV, 64)
    grid = lambda meta: (triton.cdiv(meta['V'], BV), triton.cdiv(meta['K'], BK), triton.cdiv(T, BT), B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h0 if h0 is not None else torch.tensor(0, dtype=torch.float), ht, h,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        H=H, T=T, K=K, V=V, BT=BT, BV=BV,
        num_warps=min(max(BT // 16, 1), 16) if extra_pass else 1,
