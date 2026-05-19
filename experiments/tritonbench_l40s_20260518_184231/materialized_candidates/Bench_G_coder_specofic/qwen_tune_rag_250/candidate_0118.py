, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_rwkv6_bwd_dq_kernel(
    k, v, w, u, 
    do, dq, dq_aux, 
    h0,
    s_k_h, s_v_h, scale,
    B, H, T, BK, BV, K, V, 
    USE_INITIAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    p_dq_aux = dq_aux + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)

    p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_dq = tl.zeros([BK], dtype=tl.float32)
    b_dq_aux = tl.zeros([BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    b_u = tl.exp(b_u)
    for _ in range(0, T):
        tl.store(p_dq_aux, b_dq_aux.to(p_dq_aux.dtype.element_ty), mask=mask_bv)
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_kv = b_k[None, :] * b_u[None, :]
        b_dh += b_kv * b_do[:, None]
        b_dq_aux += b_do * b_kv * b_u[None, :]
        b_dq += tl.sum(b_dh * b_u[None, :], axis=1)
        b_dh = b_dh * b_u[:, None]
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_dh *= b_w[None, :]
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_kv = b_k[None, :] * b_u[None, :]
        b_dq += tl.sum(b_dh * b_kv, axis=1)
        b_dh -= b_kv * tl.sum(b_dh * b_u[None, :], axis=1)[:, None]
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_bk)
        p_k += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_dq += -K if REVERSE else K
        p_w += -K if REVERSE else K
    b_dq += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_bk)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_h0, b_dh.to(p_h0.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_rwkv6_bwd_dkv_kernel(
    q, k, v, w, u, 
    do, dk, dk_aux, dv, dh0, 
    s_k_h, s_v_h, scale,
    B, H, T, BK, BV, K, V, 
    USE_INITIAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dk = dk + (i_bh + i_k * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dh0 = dh0 + (i_bh + i_k * B * H) * K * V + tl.arange(0, BK)[None, :] * V + (i_v * BV + tl.arange(0, BV)[:, None]) + i_k * K * V

    p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_dk = tl.zeros([BK, BV], dtype=tl.float32)
    b_dv = tl.zeros([BK, BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    b_u = tl.exp(b_u)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_kv = b_k[:, None] * b_v[None, :]
        b_dh += b_kv * b_do[None, :] * b_u[:, None]
        b_dk += b_do[:, None] * b_v[None, :] * b_u[:, None]
