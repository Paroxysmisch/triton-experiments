import torch
import triton
import triton.language as tl

@triton.jit
def chunk_gated_abc_fwd_kernel_cum(
    s,
    o,
    s_t,
    s_s,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s = tl.arange(0, BS)
    m_s = i_s[:, None] >= i_s[None, :]

    i_t = tl.program_id(0) * BT + tl.arange(0, BT)[:, None]
    i_s = tl.program_id(1) * BS + tl.arange(0, BS)[None, :]

    p_s = tl.make_block_ptr(s + i_t * S, (T, S), (S, 1), (i_t[0], i_s[0]), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_t * S, (T, S), (S, 1), (i_t[0], i_s[0]), (BT, BS), (1, 0))

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

@triton.jit
def chunk_gated_abc_fwd_kernel_h(
    k,
    v,
    g,
    h0,
    ht,
    h,
    s_kt,
    s_kh,
    s_kd,
    s_vt,
    s_vh,
    s_vd,
    s_gt,
    s_gh,
    s_gd,
    s_ht,
    s_hh,
    s_hd,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NH: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_k = tl.program_id(0) * BK + tl.arange(0, BK)[:, None]
    i_v = tl.program_id(1) * BV + tl.arange(0, BV)[None, :]
    i_bh = tl.program_id(2)

    i_t = tl.arange(0, BT)[None, :]

    p_k = tl.make_block_ptr(k + i_t * K, (T, K), (1, K), (i_t[0], i_k[0]), (BT, BK), (0, 1))
    p_g = tl.make_block_ptr(g + i_t * K, (T, K), (1, K), (i_t[0], i_k[0]), (BT, BK), (0, 1))
    p_h = tl.make_block_ptr(h + (i_t * H + i_bh * NT * H)[:, None], (T, H), (H, 1), (i_t[:, None], i_bh), (BT, 1), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_h = tl.zeros([BT, BV], dtype=tl.float32)

    for _ in range(0, i_v.shape[0]):
        b_h += tl.dot(b_k.to(b_h.dtype), b_h.to(b_h.dtype), allow_tf32=False)
        b_h *= (1 - b_g)
        b_g = tl.where(tl.arange(0, BT)[:, None] == 0, 1.0, b_g)

        p_h = tl.make_block_ptr(
            h + (i_t * H + i_bh * NT * H)[:, None],
            (T, H),
            (H, 1),
            (i_t[:, None], i_bh),
            (BT, 1),
            (1, 0),
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        i_v = tl.roll(i_v, -1)
        i_bh = tl.roll(i_bh, -1)
        b_h = tl.roll(b_h, -1, axis=0)

    if USE_INITIAL_STATE:
        i_ht = tl.arange(0, BT)[:, None] + (T - 1) * BT + i_bh * BT
        p_ht = tl.make_block_ptr(ht + i_ht, (BT, H), (H, 1), (0, i_bh), (BV, 1), (1, 0))
        b_ht = tl.load(p_ht, boundary_check=(0, 1))
        b_h = b_h * 0 + b_ht

    if STORE_FINAL_STATE:
        i_h0 = tl.arange(0, BT)[:, None] + i_bh * BT
        p_h0 = tl.make_block_ptr(h0 + i_h0, (BT, H), (H, 1), (0, i_bh), (BV, 1), (1, 0))
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), boundary_check=(0, 1))

def fwd_pre(g, B, H, T, K):
    g = g.contiguous()
    s = torch.empty((B, H, T, K), dtype=g.dtype, device=g.device)
    o = torch.empty_like(s)

    NT = triton.cdiv(T, K)
    BS = 64
    BT = min(128, triton.next_power_of_2(K))

    def grid(meta):
        return (triton.cdiv(meta["T"], meta["BT"]), BS, 1)

    chunk_gated_abc_fwd_kernel_cum[grid](
        s,
        o,
        s.stride(2),
        s.stride(1),
        T=NT * K,
        S=K,
        BT=BT,
        BS=BS,
    )

    g = g.unsqueeze(2)
    o = o.unsqueeze(2)

    return g, o, NT, K

def fwd_inner(
    k,
    v,
    g,
    h0,
    ht,
    B,
    H,
    T,
    K,
    V,
    NT,
    BK,
    BV,
    ch,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
):
    K = K // ch
    V = V // ch

    NT = triton.cdiv(T, K)
    BK = min(64, triton.next_power_of_2(K))
    BV = min(64, triton.next_power_of_2(V))
    BS = 1

    BT = min(128, triton.next_power_of_2(K))

    if BV % BS != 0:
        BV = triton.next_power_of_2(BV)

    NT = triton.cdiv(T, BT)

    h = torch.empty(
        (B, H, T, K // ch), dtype=k.dtype, device=torch.cuda.current_device()
    )

    grid = lambda META: (
        NT,
        triton.cdiv(K, META["BK"]),
        triton.cdiv(V, META["BV"]),
    )

    def chunk_gated_abc_fwd_kernel_h_1(
        k,
        v,
        g,
        h,
        s_kt,
        s_kh,
        s_kd,
        s_vt,
        s_vh,
        s_vd,
        s_gt,
        s_gh,
        s_gd,
        s_ht,
        s_hh,
        s_hd,
        T: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        H: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        NT: tl.constexpr,
        NH: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
    ):
        return chunk_gated_abc_fwd_kernel_h[grid](
            k,
            v,
            g,
            h0,
            ht,
            h,
            s_kt,
            s_kh,
            s_kd,
            s_vt,
            s_vh,
            s_vd,
            s_gt,
            s_gh,
            s_gd,
            s_ht,
            s_hh,
            s_hd,
            T=T,
            K=K,
            V=V,
            H=H,
            BT=BT,
            BK=BK,
            BV=BV,
            NT=NT,
            NH=H,
            USE_INITIAL_STATE=USE_INITIAL_STATE,
            STORE_FINAL_STATE=STORE_FINAL_STATE,
        )

    chunk_gated_abc_fwd_kernel_h_1[grid](
        k,
        v,
        g,
        h,
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        g.stride(1),
        g.stride(2),
        g.stride(3),
        ht.stride(1),
        ht.stride(2),
        ht.stride(3),
        T=T,
        K=K,
        V=V,
        H=H,
        BT=BT,
        BK=BK,
        BV=BV,
        NT
