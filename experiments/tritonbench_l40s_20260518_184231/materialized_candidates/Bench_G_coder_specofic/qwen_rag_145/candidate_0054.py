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


@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    h, g, s_s_h, s_s_t, s_s_d,
    BH: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)
    
    p_s = tl.make_block_ptr(h + i_bh * s_s_h, (BH, BT), (s_s_t, s_s_d), (i_t * BT, i_s * BV), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_s_h, (BH, BT), (s_s_t, s_s_d), (i_t * BT, i_s * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(h + i_bh * s_s_h, (BH, BT), (s_s_t, s_s_d), (i_t * BT, i_s * BV), (BT, BV), (1, 0))
    
    b_h = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_h = b_h * b_g
    
    b_o = tl.dot(m_s, b_h, allow_tf32=False)
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


def fwd_inner(h, g, BH, BT, BV):
    ht, h = h, torch.empty_like(h, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['BH'], meta['BV']), BT, BH)
    chunk_gated_abc_fwd_kernel_h[grid](
        ht, g,
        h.stride(1), h.stride(2), h.stride(3),
        BH=BH, BT=BT, BV=BV
    )
    return h
