(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * _gk[None, :]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * _gv[:, None]
        h += _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_gated_abc_bwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    do,
    dq,
    dk,
    dv,
    dh0,
    s_k_h,
    s_v_h,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    dh = tl.zeros([BK, BV], dtype=tl.float32)

    for _ in range(0, T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        dh += _k[:, None] * _v[None, :] * _do[None, :]
        _d_q = tl.sum(dh * _q[:, None], axis=1)
        tl.store(p_dq, _d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_dq += -K if REVERSE else K
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V

    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dk = dk + (i_bh + i_v * H * B) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_dv = dv + (i_bh + i_k * H * B) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    for _ in range(0, T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
        dh = dh * _do[None, :]
        if USE_GK:
            dh = dh * _gk[:, None]
        if USE_GV:
            dh = dh * _gv[None, :]
        _d_k = tl.sum(dh * _v[None, :], axis=1)
        _d_v = tl.sum(dh * _k[:, None], axis=0)
        tl.store(p_dk, _d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, _d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do += -V if REVERSE else V
        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_dk += -K if REVERSE else K
        p_dv += -V if REVERSE else V
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_dh0, dh.to(p_dh0.dtype.element_ty), mask=mask_kv)

class FusedRecurrentGatedABCFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, scale=None, initial_state=None, store_final_state=False, gk=None, gv=None):
        B, H, T, K, V = *q.shape, v.shape[-1]
        BK, BV = min(triton.next_power_of_2(K), 16), min(triton.next_power_of_2(V), 16)
        grid = (triton.cdiv(V, BV), triton.cdiv(K, BK), B * H)
        o = torch.empty_like(v)
        if store_final_state:
            ht = torch
