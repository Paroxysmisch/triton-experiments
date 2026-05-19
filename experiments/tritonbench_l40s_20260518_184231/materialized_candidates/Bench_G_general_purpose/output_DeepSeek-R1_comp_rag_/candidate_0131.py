import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_v,
    stride_out_b, stride_out_h, stride_out_m, stride_out_v,
    B, H, T, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, USE_FP8: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    bh = pid_bh
    b = bh // H
    h = bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptr = Q + b * stride_q_b + h * stride_q_h + (offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_k)
    mask_q = offs_m < T
    q = tl.load(q_ptr, mask=mask_q[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    num_n_blocks = tl.cdiv(T, BLOCK_N)
    
    for n in range(num_n_blocks):
        offs_n = n * BLOCK_N + offs_n_base

        k_ptr = K + b * stride_k_b + h * stride_k_h + (offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_k)
        mask_k = offs_n < T
        k = tl.load(k_ptr, mask=mask_k[:, None], other=0.0)
        
        v_ptr = V + b * stride_v_b + h * stride_v_h + (offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_v)
        mask_v = offs_n < T
        v = tl.load(v_ptr, mask=mask_v[:, None], other=0.0)

        if USE_FP8:
            k = k.to(tl.float8e5)
            v = v.to(tl.float8e5)

        k = k.to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        if IS_CAUSAL:
            mask = (offs_m[:, None] >= offs_n[None, :])
            s = tl.where(mask, s, float('-inf'))

        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(s - m_curr[:, None])
        
        l_curr = alpha * l_i + tl.sum(p, axis=1)
        p_scale = p / l_curr[:, None]
        
        acc = acc * alpha[:, None] + tl.dot(p_scale.to(v.dtype), v.to(v.dtype))
        m_i = m_curr
        l_i = l_curr

    out_ptr = Out + b * stride_out_b + h * stride_out_h + (offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_v)
    tl.store(out_ptr, acc.to(out_ptr.dtype.element_ty), mask=mask_q[:, None])


def triton_fa(q, k, v, sm_scale, is_causal=False):
    assert q.dtype in [torch.float16, torch.bfloat16], "Queries must be fp16/bf16"
    use_fp8 = k.dtype == torch.int8 and v.dtype == torch.int8
    if use_fp8:
        assert k.dtype == torch.int8 and v.dtype == torch.int8, "FP8 requires int8 keys/values"
    else:
        assert k.dtype == q.dtype and v.dtype == q.dtype, "Data type mismatch"

    B, H, T, D = q.shape
    o = torch.empty_like(v)
    
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = D if D <= 64 else 64

    grid = (triton.cdiv(T, BLOCK_M), B * H)

    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, T, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal, USE_FP8=use_fp8,
        num_warps=4 if BLOCK_DMODEL <= 64 else 8,
        num_stages=2
    )
    return o
