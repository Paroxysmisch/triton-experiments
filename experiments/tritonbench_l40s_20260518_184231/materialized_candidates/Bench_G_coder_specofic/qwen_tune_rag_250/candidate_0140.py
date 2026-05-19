.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * DK + i_k * BK + tl.arange(0, BK)
    cum_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK
                       + i_k * BK + tl.arange(0, BK)).to(tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        _q *= tl.math.exp2(_g) * scale
        _k *= tl.math.exp2(cum_decay - _g)
        tl.store(p_qg, _q.to(p_qg.dtype.element_ty), mask=mask)
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        p_q += DK
        p_g += DK
        p_k += DK
        p_qg += DK
        p_kg += DK
    p_kg -= DK
    p_qg -= DK
    for i in range(BT):
        _q = tl.load(p_qg, mask=mask, other=0)
        _k = tl.load(p_kg, mask=mask, other=0)
        _k *= tl.math.exp2(tl.math.min(cum_decay - tl.load(p_g).to(tl.float32), 0))
        tl.store(p_kg, _k.to(p_kg.dtype.element_ty), mask=mask)
        p_qg += DK
        p_kg += DK
        p_g += DK

# Kernel for backward decay global cumsum
@triton.jit
def bwd_decay_global_cumsum(
    dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg,
    s_qk_h, s_qk_t, s_qk_d, B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, DK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * DK
    cum_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * DK * DK +
                       i_k * BK + tl.arange(0, BK)).to(tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < DK
    mask_load_g = (i_k * BK + tl.arange(0, BK)) < DK

    for i in range(BT):
        _dq_inner = tl.load(p_dq_inner, mask=mask, other=0)
        _dq_inter = tl.load(p_dq_inter, mask=mask, other=0)
        _dk_inner = tl.load(p_dk_inner, mask=mask, other=0)
        _dk_inter = tl.load(p_dk_inter, mask=mask, other=0)
        _q = tl.load(p_q, mask=mask, other=0)
        _k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask_load_g, other=0).to(tl.float32)
        _dg = _dq_inner * tl.math.exp2(_g) * scale + _dq_inter * tl.math.exp2(cum_decay - _g)
        tl.store(p_dg, _dg.to(p_dg.dtype.element_ty), mask=mask)
        _dk = _dk_inter * tl.math.exp2(cum_decay - _g)
        _dk += _dk_inner * tl.math.exp2(tl.math.min(cum_decay - _g, 0))
        tl.store(p_dk, _dk.to(p_dk.dtype.element_ty), mask=mask)
        p_dq_inner -= DK
        p_dk_inner -= DK
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
    p_dg += DK
    for i in range(BT):
        _dq_inter = tl.load(p_dq_inter, mask=mask, other=0)
        _dk_inter = tl.load(p_dk_inter, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask_load_g, other=0).to(tl.float32)
        _dg = _dq_inter * tl.math.exp2(cum_decay - _g)
        tl.store(p_dg, _dg.to(p_dg.dtype.element_ty), mask=mask)
        _dk = _dk_inter * tl.math.exp2(tl.math.min(cum_decay - _g, 0))
        tl.store(p_dk, _dk.to(p_dk.dtype.element_ty), mask=mask)
        p_dq_inter -= DK
        p_dk_inter -= DK
        p_dg -= DK
        p_dk -= DK

def fwd_decay_cumsum_launcher(q, g, B, H, T, scale):
    DK = q.shape[-1]
    BT = 64
    BK = min(64, triton.next_power_of_2(DK))
    num_warps = 4
    if DK <= 128:
        num_warps = 8
    if DK <= 64:
        num_warps = 16
    grid = (triton.cdiv(DK, BK), triton.cdiv(T, BT), B * H)
    s_qk = q.strides
    fwd_decay_cumsum[grid](q, g, s_qk[0], s_qk[1], s_qk[2], B, H, T, scale, BT=BT, BK=BK, DK=DK, num_warps=num_warps)

def prepare_qg_kg_launcher(q, k, g, qg, kg, B, H, T, scale):
    DK = q.shape[-1]
    BT = 64
    BK = min(64, triton.next_power_of_2(DK))
    num_warps = 4
    if DK <= 128:
        num_warps = 8
    if DK <= 64:
        num_warps = 16
    grid = (triton.cdiv(DK, BK), triton.cdiv(T, BT), B * H)
    s_qk = q.strides
    prepare_qg_kg[grid](q, k, g, qg, kg, s_qk[0], s_qk[1], s_qk[2], B, H, T, scale, BT=BT, BK=BK, DK=DK, num_warps=num_warps)

def bwd_decay_global_cumsum_launcher(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, B, H, T, scale):
    DK = q.shape[-1]
    BT = 64
    BK = min(64, triton
