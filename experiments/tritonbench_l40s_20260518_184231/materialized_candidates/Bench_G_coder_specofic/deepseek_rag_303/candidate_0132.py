import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, os, 
    Lk, Lv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, 
    m_size, head_size, batch, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr, 
    IS_CAUSAL: tl.constexpr, USE_FP8: tl.constexpr
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    if USE_FP8:
        Q = Q.to(tl.int8)
        K = K.to(tl.int8)
        V = V.to(tl.int8)
    off_d = tl.arange(0, BLOCK_DMODEL)
    remat_off_k = tl.arange(0, BLOCK_N) % Lk
    remat_off_v = tl.arange(0, BLOCK_N) % Lv
    
    off_q = off_h * s_qk_h + (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * \
        s_qk_t + off_d[None, :]
    off_k = off_h * s_qk_h + (remat_off_k[None, :] + tl.arange(0, BLOCK_M))[None, :]\
         * s_qk_t + off_d[:, None] + tl.arange(0, BLOCK_N)[None, :] * s_qk_d
    off_v = off_h * s_vo_h + (remat_off_v[:, None] + tl.arange(0, BLOCK_M)[None, :]) *\
         s_vo_t + off_d[None, :]
    logits = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    max_logit = tl.zeros([2], dtype=tl.float32)
    q = tl.load(Q + off_q, mask=off_q < (head_size) * batch * BLOCK_DMODEL, other=0.0)
    for start_n in range(0, tl.cdiv(m_size, BLOCK_N)):
        k = tl.load(K + off_k, mask=off_k < m_size * BLOCK_DMODEL, other=0.0)
        v = tl.load(V + off_v, mask=off_v < m_size * BLOCK_DMODEL, other=0.0)
        qk = tl.dot(q, k)

        if USE_FP8:
            qk = qk.to(tl.bfloat16)
        logits = tl.where((start_n * BLOCK_N + off_n)[:, None] < m_size, qk, - float(
            "inf") if not IS_CAUSAL else 0.) * sm_scale
        max_logit = tl.maximum(tl.max(logits, 1), max_logit)
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N + IS_CAUSAL)
        tl.device_assert((off_n[:, None] < m_size) & \
            (off_n[None, :] < m_size), "start_n " + str(start_n))
        if not IS_CAUSAL:
            causal_mask = (off_n[:, None] >= off_n[None, :]).to(tl.float32)
        else:
            causal_mask = (off_n[:, None] > off_n[None, :]).to(tl.float32)
        logits += causal_mask * (-float("inf"))
        start_n += 1
    x = tl.softmax(logits)
    out = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, tl.cdiv(m_size, BLOCK_N)):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        mask = (off_n[:, :] < m_size)[:, None] & ((start_m * BLOCK_M + \
                tl.arange(0, BLOCK_M)[None, :]) < m_size)
        v = tl.load(V + off_v, mask=off_v < m_size, other=0.0).to(tl.float32)

        if USE_FP8:
            v = v.to(tl.bfloat16)
        out += tl.dot(x, v)
        off_v = off_h * s_vo_h + (off_n + tl.arange(0, BLOCK_M)) * s_vo_t + \
            off_d[None, :]
    off_o = (off_h * batch + start_m) * BLOCK_DMODEL + \
        tl.arange(0, BLOCK_DMODEL)[None, :] + s_vo_d
    mask = off_o < head_size * batch * BLOCK_DMODEL
    tl.store(os + off_o, out, mask=mask)


@custom_fwd(cast_inputs=torch.bfloat16)
def triton_fa(
    q, k, v, sm_scale, BLOCK_M: int = 64, BLOCK_N: int = 128,
    BLOCK_DMODEL: int = 128, IS_CAUSAL: bool = False,
    return_z: bool = False, USE_FP8: bool = False
):
    assert q.dtype == k.dtype == v.dtype
    assert BLOCK_N <= 2048
    if not IS_CAUSAL:
        assert BLOCK_M == 64
    m_size = q.shape[2]
    head_size = q.shape[1]
    batch = q.shape[0]
    num_warps = 4 if BLOCK_DMODEL <= 128 else 8
    zero = q.new_tensor(0.)
    out = torch.empty_like(q)
    if use_fwd_hook:
        k, v = fwd_hook(k, v)
    grid = (triton.cdiv(m_size, BLOCK_M), head_size * \
         batch)
    dtype = q.dtype
    if USE_FP8:
        q = q.to(torch.int8)

    _fwd_kernel[grid](
        q, k, v, sm_scale, out, k.stride(1), v.stride(1), k.stride(2),
        k.stride(3), k.stride(4), v.stride(2), v.stride(3), v.stride(4), m_size,
        head_size, batch, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, IS_CAUSAL=IS_CAUSAL,
        num_warps=num_warps, use_fp8=USE_FP8
    )
    
        
    return out, zero
