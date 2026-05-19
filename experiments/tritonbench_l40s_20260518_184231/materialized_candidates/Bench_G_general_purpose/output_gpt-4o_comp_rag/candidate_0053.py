import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k, v, g, h0, ht, h,
    k_s_h, k_s_t, k_s_d,
    v_s_h, v_s_t, v_s_d,
    g_s_h, g_s_t, g_s_d,
    h_s_h, h_s_t, h_s_d,
    T: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_h, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BK)
    m_k = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    p_k = tl.make_block_ptr(k + i_bh * k_s_h, (T, BK), (k_s_t, k_s_d), (i_t * BK, i_h * BK), (BK, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * v_s_h, (T, BV), (v_s_t, v_s_d), (i_t * BV, i_h * BV), (BV, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * g_s_h, (T, BK), (g_s_t, g_s_d), (i_t * BK, i_h * BK), (BK, BK), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * h_s_h, (T, BV), (h_s_t, h_s_d), (i_t * BV, i_h * BV), (BV, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    gated_v = b_g * b_v
    b_h = tl.dot(m_k, gated_v, allow_tf32=False)

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

def fwd_inner(k, v, g, h0, ht, h, B, H, T, K, V, BK, BV):
    NT = triton.cdiv(T, BK)
    def grid(meta): return (triton.cdiv(meta['K'], meta['BK']), NT, B * H)
    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h0, ht, h,
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        g.stride(1), g.stride(2), g.stride(3),
        h.stride(1), h.stride(2), h.stride(3),
        T=T, BK=BK, BV=BV
    )
    return h
