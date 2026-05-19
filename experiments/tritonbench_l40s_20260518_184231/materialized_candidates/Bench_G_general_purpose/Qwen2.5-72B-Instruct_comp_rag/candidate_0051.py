import torch
import triton
import triton.language as tl

# Kernel for cumulative operation
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

# Kernel for gated cumulative sum
@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h0, ht, h, k_s_h, k_s_t, k_s_d, v_s_h, v_s_t, v_s_d, g_s_h, g_s_t, g_s_d, h_s_h, h_s_t, h_s_d,
    T: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_k = tl.make_block_ptr(k + i_bh * k_s_h, (T, S), (k_s_t, k_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * v_s_h, (T, S), (v_s_t, v_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * g_s_h, (T, S), (g_s_t, g_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * h_s_h, (T, S), (h_s_t, h_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    b_h = tl.zeros((BT, BS), dtype=tl.float32)
    if i_t == 0:
        b_h0 = tl.load(h0 + i_bh * h_s_h, boundary_check=(0, 1)).to(tl.float32)
        b_h = b_h0

    for t in range(i_t * BT, (i_t + 1) * BT):
        b_h = b_h + b_k * b_v * b_g
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    if i_t == (T // BT - 1):
        tl.store(ht + i_bh * h_s_h, b_h.to(ht.dtype.element_ty), boundary_check=(0, 1))

# Wrapper function for cumulative operation
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

# Wrapper function for gated cumulative sum
def fwd_inner(k, v, g, h0, ht, h, B, H, T, S, BT, BK, BV):
    NT = triton.cdiv(T, BT)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h0, ht, h,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        g.stride(1), g.stride(2), g.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        T=T, S=S, BT=BT, BK=BK, BV=BV
    )
