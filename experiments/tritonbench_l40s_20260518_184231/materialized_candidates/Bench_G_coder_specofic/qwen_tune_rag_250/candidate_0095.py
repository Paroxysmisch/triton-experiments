_o = h * _q[:, None]

        o_i = tl.sum(_o, axis=1)
        tl.store(p_o, o_i.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)


@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.pow(2, -5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        p_q_ = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
        _q = tl.load(p_q_, mask=mask_bk, other=0).to(tl.float32) * scale

        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        h = b_b * h + _q[:, None] * _do[None, :]
        _dq = h * _do[:, None]
        tl.store(p_dq, _dq.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_dq += DK
        p_do += DV
        p_q_ += DK

    p_k += (T - 1) * DK
    p_v += (T - 1) * DV
    p_dk += (T - 1) * DK
    p_dv += (T - 1) * DV

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        _dv = h * _do[None, :]
        tl.store(p_dv, _dv.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV
        p_do -= DV
        p_dv -= DV

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)

    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h += _k[:, None] * _v[None, :]
        _dq = h * _do[None, :]
        tl.store(p_dq, _dq.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += DK
        p_do += DV
        p_dq += DK
        p_v += DV

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)

    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h += _k[:, None] * _v[None, :]
        _dv = h * _do[None, :]
        tl.store(p_dv, _dv.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do += DV
        p_dv += DV
        p_k += DK
        p_v += DV


def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
):
    B, H, T, DK = q.shape
    _, _, _, DV = v.shape

    BK = min(128, triton.next_power_of_2(DK))
    BV = min(128, triton.next_power_of_2(DV))

    o = torch.empty(B, H, T, DV, dtype=q.dtype, device=q.device)
    if output_final_state:
        final_state = torch.empty(B, H, DK, DV, dtype=torch.float32,
                                   device=q.device)
    else:
        final_state = None

    scale = DK ** -0.5

    def grid(meta): return (triton.cdiv(DV, meta['BV']), triton.cdiv(DK, meta['BK']),
                            B * H)
    fused_recurrent_retention_fwd_kernel[grid](
        q, k, v, o, initial_state, final_state,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        B, H, T, scale,
        BK=BK, BV=BV, DK=DK, DV=DV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
    )

    return o, final_state if output_final_state else None


def fused_recurrent_retention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
):
    B, H, T, DK = q.shape
    _, _, _, DV = v.shape

    BK = min(128, triton.next_power_of_2(DK))
    BV = min(128
