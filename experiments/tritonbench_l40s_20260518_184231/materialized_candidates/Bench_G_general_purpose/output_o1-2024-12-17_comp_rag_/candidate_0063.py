import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def fused_recurrent_gated_abc_fwd_kernel(
    q,
    k,
    v,
    gk,
    gv,
    o,
    h0,
    ht,
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
    STORE_FINAL_STATE: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
):
    # indices
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)

    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V

    h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * b_gk[None, :]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * b_gv[:, None]
        h += b_k[None, :] * b_v[:, None]
        b_o = h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
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
    h0,
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
    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * b_gk[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * b_gv[None, :]
        h += b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_q += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_dq += -K if REVERSE else K
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V

    # sync threads
    tl.debug_barrier()

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            b_dh *= b_gk[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            b_dh *= b_gv[None, :]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dv += V if REVERSE else -V
        if USE_GK:
            p_gk += K if REVERSE else -K
        if USE_GV:
            p_gv += V if REVERSE else -V


class FusedRecurrentGatedABCFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, s, g, scale=None, initial_state=None, output_final_state=False, reverse=False):
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        # default scale
        if scale is None:
            scale = K ** -0.5

        BK, BV, BM = min(K, 32), min(V, 32), min(M, 32)
        NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)
        num_stages = 1
        num_warps = 1

        g = g.float().exp()

        final_state = (None, None)
        if output_final_state:
            final_state = (q.new_empty(B, H, K, M), q.new_empty(B, H, M, V))

        ok = q.new_empty(NK, B, H, T, M, dtype=torch.float)
        gk, gv = None, g
        grid = (NM, NK, B * H)
        fused_recurrent_gated_abc_fwd_kernel[grid](
            q, k, s, gk, gv, ok, initial_state[0], final_state[0],
            k.stride(1),
            s.stride(1),
            scale=scale,
            B=B, H=H, T=T, K=K, V=M, BK=BK, BV=BM,
            USE_INITIAL_STATE=initial_state[0] is not None,
            STORE_FINAL_STATE=final_state[0] is not None,
            USE_GK=False,
            USE_GV=True,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ok = ok.sum(0)

        qv = ok.softmax(-1, dtype=torch.float)
        ov = q.new_empty(NM, B, H, T, V, dtype=torch.float)
        gk, gv = g, None
        grid = (NV, NM, B * H)
        fused_recurrent_gated_abc_fwd_kernel[grid](
            qv, s, v, gk, gv, ov, initial_state[1], final_state[1],
            s.stride(1),
            v.stride(1),
            scale=1.,
            B=B, H=H, T=T, K=M, V=V, BK=BM, BV=BV,
            USE_INITIAL_STATE=initial_state[0] is not None,
            STORE_FINAL_STATE=final_state[0] is not None,
            USE_GK=True,
            USE_GV=False,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )
        ov = ov.sum(0)

        ctx.save_for_backward(q, k, v, s, g, qv, *initial_state, ok)
        ctx.scale = scale
        ctx.reverse = reverse
        # we do not need the gradient of the final state from the next chunk
        # similiar to Trunctated BPTT
        if final_state is not None:
            final_state = tuple(i.detach() for i in final_state)
        return ov.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, s, g, qv, *initial_state, ok = ctx.saved_tensors
        B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
        V = v.shape[-1]
        scale = ctx.scale

        BK, BV, BM = min(K, 32), min(V, 32), min(M, 32)
        NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), tr
