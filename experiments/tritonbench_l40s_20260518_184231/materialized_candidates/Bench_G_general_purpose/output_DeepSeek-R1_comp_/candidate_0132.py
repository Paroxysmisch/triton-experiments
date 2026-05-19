import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_d,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_d,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_d,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_d,
    batch_size, num_heads, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr, USE_FP8: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Range of queries handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q pointer
    q_ptr = Q + pid_batch * stride_q_batch + pid_head * stride_q_head + offs_m[:, None] * stride_q_seq + offs_d[None, :] * stride_q_d
    mask_q = (offs_m < seq_len)[:, None] & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptr, mask=mask_q, other=0.0)
    q = q * sm_scale  # Apply scaling immediately

    # Initialize O, m, l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K and V
    lo = 0
    hi = tl.cdiv(seq_len, BLOCK_N) if not IS_CAUSAL else tl.minimum((pid_m * BLOCK_M + BLOCK_M + BLOCK_N - 1) // BLOCK_N, tl.cdiv(seq_len, BLOCK_N))
    for n in range(lo, hi):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Load K
        k_ptr = K + pid_batch * stride_k_batch + pid_head * stride_k_head + offs_n[:, None] * stride_k_seq + offs_d[None, :] * stride_k_d
        mask_k = (offs_n < seq_len)[:, None] & (offs_d[None, :] < BLOCK_DMODEL)
        k = tl.load(k_ptr, mask=mask_k, other=0.0)
        if USE_FP8:
            k = k.to(tl.float8e5, bitcast=True).to(tl.float16)
        
        # Compute QK
        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32)
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n[None, :])
            qk = qk * causal_mask + (-float('inf')) * (1 - causal_mask)
        
        # Compute m, l, O updates
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        alpha = tl.exp(m_prev - m_curr)
        beta = tl.exp(qk - m_curr[:, None])
        l_curr = alpha * l_prev + tl.sum(beta, 1)
        
        # Load V
        v_ptr = V + pid_batch * stride_v_batch + pid_head * stride_v_head + offs_n[:, None] * stride_v_seq + offs_d[None, :] * stride_v_d
        v = tl.load(v_ptr, mask=mask_k, other=0.0)
        if USE_FP8:
            v = v.to(tl.float8e5, bitcast=True).to(tl.float16)
        
        # Update acc
        acc = acc * alpha[:, None] + tl.dot(beta.to(q.dtype), v.to(q.dtype))
        acc = acc.to(tl.float32)
        
        m_prev = m_curr
        l_prev = l_curr

    # Write output
    acc = acc / l_prev[:, None]
    out_ptr = Out + pid_batch * stride_out_batch + pid_head * stride_out_head + offs_m[:, None] * stride_out_seq + offs_d[None, :] * stride_out_d
    tl.store(out_ptr, acc.to(Out.type.element_ty), mask=mask_q)

def triton_fa(q, k, v, sm_scale, causal=False, use_fp8=False):
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Inputs must be 4D: (batch, heads, seq, dim)"
    batch, heads, seq_len, d_model = q.shape
    assert d_model in {16, 32, 64, 128}, "d_model must be 16, 32, 64, or 128"
    if use_fp8:
        assert k.dtype == torch.int8 and v.dtype == torch.int8, "K/V must be int8 for FP8 mode"
    
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64  # Tune based on hardware
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))
    num_warps = 4 if d_model <= 64 else 8
    
    _fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d_model,
        IS_CAUSAL=causal, USE_FP8=use_fp8,
        num_warps=num_warps, num_stages=2
    )
    return out
