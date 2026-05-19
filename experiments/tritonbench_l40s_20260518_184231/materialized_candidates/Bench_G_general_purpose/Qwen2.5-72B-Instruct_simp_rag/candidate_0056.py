import torch
import triton
import triton.language as tl

# Cumulative Kernel
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

# Gated Accumulation Kernel
@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h, h0, ht, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_g_h, s_g_t, s_g_d, s_h_h, s_h_t, s_h_d,
    T: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr
):
    i_h, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, S), (s_k_t, s_k_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, S), (s_v_t, s_v_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, S), (s_g_t, s_g_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, S), (s_h_t, s_h_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    b_h = tl.zeros((BT, BS), dtype=tl.float32)
    if h0 is not None:
        p_h0 = tl.make_block_ptr(h0 + i_bh * s_h_h, (T, S), (s_h_t, s_h_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))
        b_h0 = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        b_h += b_h0

    for i in range(T):
        b_h += b_g * tl.dot(b_k, b_v, allow_tf32=False)

    if ht is not None:
        p_ht = tl.make_block_ptr(ht + i_bh * s_h_h, (T, S), (s_h_t, s_h_d), (i_t * BT, i_h * BS), (BT, BS), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

# Wrapper for Cumulative Kernel
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

# Wrapper for Gated Accumulation Kernel
def fwd_inner(k, v, g, h, h0=None, ht=None, B, H, T, S, BT, BS, BD):
    NT = triton.cdiv(T, BT)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h, h0, ht,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        g.stride(1), g.stride(2), g.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        T=T, S=S, BT=BT, BS=BS, BD=BD
    )
    return h
