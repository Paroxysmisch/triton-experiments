import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    L, M,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    stride_l, stride_m,
    BATCH, M_Q, N_K, D_MODEL,
    sm_scale,
    USE_MASK: tl.constexpr,
    row_offs, col_offs, **meta
):
    b_idx = tl.program_id(0)
    # Q, K, V block offsets
    q_offs = b_idx * stride_qh
    k_offs = b_idx * stride_kh
    v_offs = b_idx * stride_vh
    o_offs = b_idx * stride_oh

    row_idx = row_offs + tl.arange(0, BLOCK_M)
    col_idx = col_offs + tl.arange(0, BLOCK_N)

    # Pointers for Q, K, V
    q_ptrs = Q + q_offs + (row_idx[:, None] * stride_qm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
    k_ptrs = K + k_offs + (col_idx[None, :] * stride_km) + (tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd)
    v_ptrs = V + v_offs + (col_idx[:, None] * stride_vm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd)

    # Load small blocks of Q and K
    q_tile = tl.load(q_ptrs, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    k_tile = tl.load(k_ptrs, mask=(col_idx[None, :] < N_K) & (tl.arange(0, BLOCK_DMODEL)[:, None] < D_MODEL), other=0.0)
    # Compute attention logits
    qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d in range(0, D_MODEL, BLOCK_DMODEL):
        q_part = q_tile
        k_part = k_tile
        qk += tl.dot(q_part, k_part)
    qk *= sm_scale

    if USE_MASK:
        mask = (row_idx[:, None] >= col_idx[None, :])
        qk = tl.where(mask, qk, float('-inf'))

    # Compute max for numerical stability
    m_i = tl.max(qk, 1)
    tl.store(M + b_idx * stride_m + row_idx, m_i, mask=(row_idx < M_Q))
    # Subtract max and exponentiate
    qk_minus_m = qk - m_i[:, None]
    numerator = tl.exp(qk_minus_m)
    l_i = tl.sum(numerator, 1)
    tl.store(L + b_idx * stride_l + row_idx, l_i, mask=(row_idx < M_Q))
    # Normalize
    denom = 1.0 / l_i
    p = numerator * denom[:, None]
    # Multiply p by V
    v_tile = tl.load(v_ptrs, mask=(col_idx[:, None] < N_K) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    out = tl.dot(p, v_tile)
    # Store output
    out_ptrs = Out + o_offs + (row_idx[:, None] * stride_om) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od)
    tl.store(out_ptrs, out, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL))

@triton.jit
def _bwd_preprocess(
    DO, L, Out, Delta,
    stride_dom, stride_doh, stride_dod,
    stride_l, stride_outm, stride_outh, stride_outd,
    stride_delta,
    BATCH, M_Q, D_MODEL, **meta
):
    b_idx = tl.program_id(0)
    row_idx = tl.arange(0, BLOCK_M) + b_idx * BLOCK_M
    mask = row_idx < M_Q

    do_ptrs = DO + (row_idx[:, None] * stride_dom) + (tl.arange(0, D_MODEL)[None, :] * stride_dod)
    out_ptrs = Out + (row_idx[:, None] * stride_outm) + (tl.arange(0, D_MODEL)[None, :] * stride_outd)
    l_ptrs = L + row_idx

    do_val = tl.load(do_ptrs, mask=(mask[:, None] & (tl.arange(0, D_MODEL)[None, :] < D_MODEL)), other=0.0)
    out_val = tl.load(out_ptrs, mask=(mask[:, None] & (tl.arange(0, D_MODEL)[None, :] < D_MODEL)), other=0.0)
    l_val = tl.load(l_ptrs, mask=mask, other=1.0)

    # Dot for delta
    delta_val = tl.sum(do_val * out_val, 1)
    tl.store(Delta + row_idx * stride_delta, delta_val, mask=mask)

@triton.jit
def _bwd_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    L, M, Delta,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_dom, stride_doh, stride_dod,
    stride_dqm, stride_dqh, stride_dqd,
    stride_dkm, stride_dkh, stride_dkd,
    stride_dvm, stride_dvh, stride_dvd,
    stride_l, stride_m, stride_delta,
    BATCH, M_Q, N_K, D_MODEL,
    sm_scale,
    USE_MASK: tl.constexpr,
    row_offs, col_offs, **meta
):
    b_idx = tl.program_id(0)
    q_offs = b_idx * stride_qh
    k_offs = b_idx * stride_kh
    v_offs = b_idx * stride_vh
    do_offs = b_idx * stride_doh
    dq_offs = b_idx * stride_dqh
    dk_offs = b_idx * stride_dkh
    dv_offs = b_idx * stride_dvh

    row_idx = row_offs + tl.arange(0, BLOCK_M)
    col_idx = col_offs + tl.arange(0, BLOCK_N)

    q_ptrs = Q + q_offs + (row_idx[:, None] * stride_qm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
    k_ptrs = K + k_offs + (col_idx[None, :] * stride_km) + (tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd)
    v_ptrs = V + v_offs + (col_idx[:, None] * stride_vm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vd)

    do_ptrs = DO + do_offs + (row_idx[:, None] * stride_dom) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_dod)
    dq_ptrs = DQ + dq_offs + (row_idx[:, None] * stride_dqm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_dqd)
    dk_ptrs = DK + dk_offs + (col_idx[None, :] * stride_dkm) + (tl.arange(0, BLOCK_DMODEL)[:, None] * stride_dkd)
    dv_ptrs = DV + dv_offs + (col_idx[:, None] * stride_dvm) + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_dvd)

    m_i = tl.load(M + b_idx * stride_m + row_idx, mask=(row_idx < M_Q), other=float('-inf'))
    l_i = tl.load(L + b_idx * stride_l + row_idx, mask=(row_idx < M_Q), other=1.0)
    delta_val = tl.load(Delta + (row_idx * stride_delta), mask=(row_idx < M_Q), other=0.0)

    q_tile = tl.load(q_ptrs, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    k_tile = tl.load(k_ptrs, mask=(col_idx[None, :] < N_K) & (tl.arange(0, BLOCK_DMODEL)[:, None] < D_MODEL), other=0.0)
    v_tile = tl.load(v_ptrs, mask=(col_idx[:, None] < N_K) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    do_tile = tl.load(do_ptrs, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)

    qk = tl.dot(q_tile, k_tile)
    qk *= sm_scale
    if USE_MASK:
        mask = (row_idx[:, None] >= col_idx[None, :])
        qk = tl.where(mask, qk, float('-inf'))

    qk_minus_m = qk - m_i[:, None]
    p = tl.exp(qk_minus_m) / l_i[:, None]
    # Weighted DO
    dv_update = tl.dot(p.to(tl.float32).T, do_tile.to(tl.float32))
    dv_old = tl.load(dv_ptrs, mask=(col_idx[:, None] < N_K) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    dv_new = dv_old + dv_update
    tl.store(dv_ptrs, dv_new, mask=(col_idx[:, None] < N_K) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL))

    # Softmax gradient
    dot_do_v = tl.dot(do_tile, v_tile.T)
    dp = (dot_do_v - delta_val[:, None]) * p
    dp *= sm_scale

    dk_update = tl.dot(q_tile.to(tl.float32).T, dp.to(tl.float32))
    dk_old = tl.load(dk_ptrs, mask=(col_idx[None, :] < N_K) & (tl.arange(0, BLOCK_DMODEL)[:, None] < D_MODEL), other=0.0)
    dk_new = dk_old + dk_update
    tl.store(dk_ptrs, dk_new, mask=(col_idx[None, :] < N_K) & (tl.arange(0, BLOCK_DMODEL)[:, None] < D_MODEL))

    dq_update = tl.dot(dp.to(tl.float32), k_tile.to(tl.float32).T)
    dq_old = tl.load(dq_ptrs, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL), other=0.0)
    dq_new = dq_old + dq_update
    tl.store(dq_ptrs, dq_new, mask=(row_idx[:, None] < M_Q) & (tl.arange(0, BLOCK_DMODEL)[None, :] < D_MODEL))

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale, use_mask):
        B, M_Q, D_MODEL = Q.shape
        _, N_K, _ = K.shape
        Q_ = Q.contiguous()
        K_ = K.contiguous()
        V_ = V.contiguous()
        Out = torch.empty_like(Q_)
        L = torch.empty((B, M_Q), dtype=Q.dtype, device=Q.device)
        M_ = torch.empty((B, M_Q), dtype=Q.dtype, device=Q.device)

        grid = lambda META: (B, )

        # Forward
        num_warps = 4
        num_stages = 2
        # We tile over M_Q, N_K in blocks
        for row_offs in range(0, M_Q, BLOCK_M):
            for col_offs in range(0, N_K, BLOCK_N):
                _fwd_kernel[grid](
                    Q_, K_, V_, Out,
                    L, M_,
                    Q_.stride(1), Q_.stride(0), Q_.stride(2),
                    K_.stride(1), K_.stride(0), K_.stride(2),
                    V_.stride(1), V_.stride(0), V_.stride(2),
                    Out.stride(1), Out.stride(0), Out.stride(2),
                    L.stride(1), M_.stride(1),
                    B, M_Q, N_K, D_MODEL,
                    sm_scale,
                    use_mask,
                    row_offs, col_offs,
                    num_warps=num_warps,
                    num_stages=num_stages
                )

        ctx.save_for_backward(Q_, K_, V_, Out, L, M_)
        ctx.sm_scale = sm_scale
        ctx.use_mask = use_mask
        return Out

    @staticmethod
    def backward(ctx, dOut):
        Q_, K_, V_, Out, L, M_ = ctx.saved_tensors
        B, M_Q, D_MODEL = Q_.shape
        _, N_K, _ = K_.shape
        DQ = torch.zeros_like(Q_)
        DK = torch.zeros_like(K_)
        DV = torch.zeros_like(V_)

        # Preprocess
        Delta = torch.zeros((B, M_Q), dtype=Q_.dtype, device=Q_.device)
        grid = lambda META: ( (M_Q + BLOCK_M - 1) // BLOCK_M * B, )
        _bwd_preprocess[grid](
            dOut, L, Out, Delta,
            dOut.stride(1), dOut.stride(0), dOut.stride(2),
            L.stride(1), Out.stride(1), Out.stride(0), Out.stride(2),
            Delta.stride(1),
            B, M_Q, D_MODEL
        )

        # Main backward
        grid = lambda META: (B, )
        num_warps = 4
        num_stages = 2
        for row_offs in range(0, M_Q, BLOCK_M):
            for col_offs in range(0, N_K, BLOCK_N):
                _bwd_kernel[grid](
                    Q_, K_, V_, dOut,
                    DQ, DK, DV,
                    L, M_, Delta,
                    Q_.stride(1), Q_.stride(0), Q_.stride(2),
                    K_.stride(1), K_.stride(0), K_.stride(2),
                    V_.stride(1), V_.stride(0), V_.stride(2),
                    dOut.stride(1), dOut.stride(0), dOut.stride(2),
                    DQ.stride(1), DQ.stride(0), DQ.stride(2),
                    DK.stride(1), DK.stride(0), DK.stride(2),
                    DV.stride(1), DV.stride(0), DV.stride(2),
                    L.stride(1), M_.stride(1), Delta.stride(1),
                    B, M_Q, N_K, D_MODEL,
                    ctx.sm_scale,
                    ctx.use_mask,
                    row_offs, col_offs,
                    num_warps=num_warps,
                    num_stages=num_stages
                )
        return DQ, DK, DV, None, None

def attention_triton(Q, K, V, sm_scale, use_mask=False):
    return _attention.apply(Q, K, V, sm_scale, use_mask)
