import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------
# Kernel: chunk_gated_abc_fwd_kernel_cum
# ------------------------------------------------------------------------
@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s, o,
    s_s_h, s_s_t, s_s_d,
    T: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    # Triangular mask for cumulative operation
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.).to(tl.float32)

    # Create block pointers
    p_s = tl.make_block_ptr(
        base=s + i_bh * s_s_h,
        shape=(T, S),
        strides=(s_s_t, s_s_d),
        offsets=(i_t * BT, i_s * BS),
        block_shape=(BT, BS),
        order=(1, 0)
    )
    p_o = tl.make_block_ptr(
        base=o + i_bh * s_s_h,
        shape=(T, S),
        strides=(s_s_t, s_s_d),
        offsets=(i_t * BT, i_s * BS),
        block_shape=(BT, BS),
        order=(1, 0)
    )

    # Load block and perform cumulative via dot with a lower-triangular mask
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)

    # Store result
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def fwd_pre(g, B, H, T, S, BT):
    NT = triton.cdiv(T, BT)
    g_org = g
    g_temp = torch.empty_like(g, dtype=torch.float)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g_temp,
        g_temp.stride(1), g_temp.stride(2), g_temp.stride(3),
        T=T, S=S, BT=BT
    )
    return g_temp

# ------------------------------------------------------------------------
# Kernel: chunk_gated_abc_fwd_kernel_h
# ------------------------------------------------------------------------
@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k_ptr, v_ptr, g_ptr,
    h0_ptr, h_ptr, ht_ptr,
    sk_h, sk_t, sk_d,
    sv_h, sv_t, sv_d,
    sg_h, sg_t, sg_d,
    sh0_h, sh0_t, sh0_d,
    sh_h, sh_t, sh_d,
    sht_h, sht_t, sht_d,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    #  i_s ~ chunk along K/V dimension, i_t ~ chunk along T dimension, i_bh ~ B*H dimension
    i_s = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)

    # Ranges for each block
    rk = tl.arange(0, BK)
    rv = tl.arange(0, BV)
    rt = tl.arange(0, BT)

    # Offsets for each block
    offs_k = (i_bh * sk_h) + (i_s * BK) * sk_d + (i_t * BT) * sk_t
    offs_v = (i_bh * sv_h) + (i_s * BV) * sv_d + (i_t * BT) * sv_t
    offs_g = (i_bh * sg_h) + (i_s * BK) * sg_d + (i_t * BT) * sg_t

    # Pointers
    p_k = tl.make_block_ptr(
        base=k_ptr + offs_k,
        shape=(T, K),
        strides=(sk_t, sk_d),
        offsets=(i_t * BT, i_s * BK),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    p_v = tl.make_block_ptr(
        base=v_ptr + offs_v,
        shape=(T, V),
        strides=(sv_t, sv_d),
        offsets=(i_t * BT, i_s * BV),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    p_g = tl.make_block_ptr(
        base=g_ptr + offs_g,
        shape=(T, K),  # g typically matches K dimension
        strides=(sg_t, sg_d),
