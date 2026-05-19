def fused_recurrent_fwd_kernel(q, k, v, o, h0, s_k_h, s_v_h, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr, BETA_SCALE: tl.constexpr):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V

    h = tl.zeros([B, H, K, V], dtype=tl.float32)
    mask_h = mask_bk[None, :, :, None] & mask_bv[:, None, None, :]

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None, None])
        h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        h += b_q[:, :, None, :] * b_k[:, :, :, None] * b_v[:, :, None, :] * BETA_SCALE
        b_o = h[:, :, :, i_v % V]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V

    if STORE_FINAL_STATE:
        p_ht = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_bwd_kernel(q, k, v, do, dq, dk, dv, h0, s_k_h, s_v_h, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, REVERSE: tl.constexpr, BETA_SCALE: tl.constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_do = do + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_k * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_bk[None, :, :, None] & mask_bv[:, None, None, :]
    h = tl.zeros([B, H, K, V], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None, :]) * V + (i_v * BV + tl.arange(0, BV)[None, :, None])
        h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_dh = b_do * b_v[:, :, None, :] * BETA_SCALE
        b_dk = tl.sum(b_dh * b_q[:, :, None, :], axis=0) * BETA_SCALE
        b_dv = tl.sum(b_dh * b_k[:, :, :, None], axis=0) * BETA_SCALE
