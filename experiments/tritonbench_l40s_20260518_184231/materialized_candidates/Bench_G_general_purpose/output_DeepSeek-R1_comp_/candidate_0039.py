import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out, sm_scale, scale_k, scale_v,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_out_b, stride_out_h, stride_out_m, stride_out_d,
    seq_len_q, seq_len_kv,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    H: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    mask_q = offs_m < seq_len_q
    q_ptr = Q + pid_batch * stride_q_b + pid_head * stride_q_h + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptr, mask=mask_q[:, None], other=0.0)
    
    k_ptr_base = K + pid_batch * stride_k_b + pid_head * stride_k_h
    v_ptr_base = V + pid_batch * stride_v_b + pid_head * stride_v_h
    
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    n_blocks = tl.cdiv(seq_len_kv, BLOCK_N)
    for n in range(n_blocks):
        n_start = n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_k = offs_n < seq_len_kv
        
        k_ptr = k_ptr_base + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d
        k = tl.load(k_ptr, mask=mask_k[:, None], other=0.0)
        k = (k.to(tl.float16) * scale_k).to(q.dtype)
        
        v_ptr = v_ptr_base + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d
        v = tl.load(v_ptr, mask=mask_k[:, None], other=0.0)
        v = (v.to(tl.float16) * scale_v).to(q.dtype)
        
        qk = tl.dot(q, tl.trans(k))
        qk *= sm_scale
        
        offs_qm = offs_m[:, None]
        offs_kn = offs_n[None, :]
        causal_mask = (offs_qm >= offs_kn) & (mask_q[:, None] & mask_k[None, :])
        qk = tl.where(causal_mask, qk, float('-inf'))
        
        m_curr = tl.maximum(tl.max(qk, axis=1), m_prev)
        alpha = tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = alpha * l_prev + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_prev, l_prev = m_curr, l_curr
    
    acc = acc / l_prev[:, None]
    out_ptr = Out + pid_batch * stride_out_b + pid_head * stride_out_h + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d
    tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=mask_q[:, None])

def context_attention_fwd_ppl_int8kv(q, k, v, o, log_factor, scale_k, scale_v):
    B, H, M, D = q.shape
    N = k.size(2)
    assert k.shape == (B, H, N, D) and v.shape == (B, H, N, D) and o.shape == (B, H, M, D), "Shape mismatch"
    
    BLOCK_M, BLOCK_N = 64, 64
    if torch.cuda.get_device_capability()[0] >= 8:
        BLOCK_M, BLOCK_N = 128, 128
    
    grid = (B, H, triton.cdiv(M, BLOCK_M))
    sm_scale = log_factor / (D ** 0.5)
    
    _fwd_kernel_int8kv[grid](
        q, k, v, o, sm_scale, scale_k, scale_v,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        M, N,
        BLOCK_DMODEL=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        H=H,
        num_warps=4,
        num_stages=4
    )
    return o
