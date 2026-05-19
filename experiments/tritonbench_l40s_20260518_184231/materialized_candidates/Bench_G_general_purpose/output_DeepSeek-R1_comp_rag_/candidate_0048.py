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
    k, v, g, h, h0, ht,
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_g_h, s_g_t, s_g_d,
    s_h_h, s_h_t, s_h_d,
    s_ht_h, s_ht_t, s_ht_d,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    HAS_INITIAL: tl.constexpr, STORE_FINAL: tl.constexpr,
    WARPS: tl.constexpr, STAGES: tl.constexpr
):
    i_k, i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2), tl.program_id(3)
    pid_t = i_t * BT + tl.arange(0, BT)
    pid_k = i_k * BK + tl.arange(0, BK)
    pid_v = i_v * BV + tl.arange(0, BV)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (pid_t[0], pid_k[0]), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (pid_t[0], pid_v[0]), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_g_h, (T, 1), (s_g_t, s_g_d), (pid_t[0], 0), (BT, 1), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_h_h, (T, V), (s_h_t, s_h_d), (pid_t[0], pid_v[0]), (BT, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_g = b_g[..., None]  # Shape (BT, 1, 1) for broadcasting

    if HAS_INITIAL:
        p_h0 = tl.make_block_ptr(h0 + i_bh * s_ht_h, (K, V), (s_ht_t, s_ht_d), (pid_k[0], pid_v[0]), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_h = tl.zeros((BK, BV), dtype=tl.float32)

    h_prev = b_h
    h_buf = tl.zeros((BT, BV), dtype=tl.float32)
    for t in tl.static_range(BT):
        k_t = b_k[t, :, None]
        v_t = b_v[t, None, :]
        g_t = b_g[t, 0, 0]
        kv = tl.dot(k_t, v_t, allow_tf32=False)
        h_curr = g_t * h_prev + (1.0 - g_t) * kv
        h_prev = h_curr
        h_buf = tl.where(tl.arange(0, BT)[:, None] == t, h_curr, h_buf)

    tl.store(p_h, h_buf.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    if STORE_FINAL:
        p_ht = tl.make_block_ptr(ht + i_bh * s_ht_h, (K, V), (s_ht_t, s_ht_d), (pid_k[0], pid_v[0]), (BK, BV), (1, 0))
        tl.store(p_ht, h_prev.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def fwd_pre(g, B, H, T, S, BT):
    NT = triton.cdiv(T, BT)
    g_org, g = g, torch.empty_like(g, dtype=torch.float)
    BS = 32  # Example block size for columns, adjust as needed
    def grid(meta): return (triton.cdiv(S, BS), NT, B * H)
    chunk_gated_abc_fwd_kernel_cum[grid](
        g_org, g,
        g_org.stride(1), g_org.stride(2), g_org.stride(3),
        T=T, S=S, BT=BT, BS=BS
    )
    return g

def fwd_inner(k, v, g, h, h0=None, ht=None, BT=64, BK=32, BV=32, WARPS=4, STAGES=2):
    B, H, T, K = k.shape
    B, H, T, V = v.shape
    has_initial = h0 is not None
    store_final = ht is not None

    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NK, NV, NT, B * H)

    s_k_h, s_k_t, s_k_d = k.stride(1), k.stride(2), k.stride(3)
    s_v_h, s_v_t, s_v_d = v.stride(1), v.stride(2), v.stride(3)
    s_g_h, s_g_t, s_g_d = g.stride(1), g.stride(2), g.stride(3)
    s_h_h, s_h_t, s_h_d = h.stride(1), h.stride(2), h.stride(3)
    s_ht_h, s_ht_t, s_ht_d = (h0.stride(1), h0.stride(2), h0.stride(3)) if has_initial else (0, 0, 0)
    if ht is not None:
        s_ht_h, s_ht_t, s_ht_d = ht.stride(1), ht.stride(2), ht.stride(3)

    chunk_gated_abc_fwd_kernel_h[grid](
        k, v, g, h, h0, ht,
        s_k_h, s_k_t, s_k_d,
        s_v_h, s_v_t, s_v_d,
        s_g_h, s_g_t, s_g_d,
        s_h_h, s_h_t, s_h_d,
        s_ht_h, s_ht_t, s_ht_d,
        T, K, V,
        BT, BK, BV,
        has_initial, store_final,
        WARPS=WARPS, STAGES=STAGES
    )
    return h, ht if store_final else None
