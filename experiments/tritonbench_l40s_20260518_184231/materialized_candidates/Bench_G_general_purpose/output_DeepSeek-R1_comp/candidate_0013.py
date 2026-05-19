import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    seqlen_q, seqlen_k, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr, KV_GROUP_NUM: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    if pid_m * BLOCK_M >= seqlen_q:
        return
    
    kv_head = pid_head // KV_GROUP_NUM
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptr = Q + pid_batch * stride_q_b + pid_head * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    mask_q = (offs_m < seqlen_q)[:, None] & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptr, mask=mask_q, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    start_n = 0
    if IS_CAUSAL:
        start_n = tl.maximum(0, (pid_m * BLOCK_M) // BLOCK_N)
    
    for pid_n in range(start_n, (seqlen_k + BLOCK_N - 1) // BLOCK_N):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        k_ptr = K + pid_batch * stride_k_b + kv_head * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        v_ptr = V + pid_batch * stride_v_b + kv_head * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
        
        mask_n = offs_n < seqlen_k
        mask_kv = mask_n[:, None] & (offs_d[None, :] < head_dim)
        
        k = tl.load(k_ptr, mask=mask_kv, other=0.0)
        v = tl.load(v_ptr, mask=mask_kv, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        
        l_i = l_i * alpha + l_ij * beta
        p_scale = beta[:, None]
        
        acc = acc * alpha[:, None] + tl.dot(p * p_scale, v.to(tl.float32))
        m_i = m_i_new

    acc = acc / l_i[:, None]
    out_ptr = Out + pid_batch * stride_out_b + pid_head * stride_out_h + offs_m[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
    tl.store(out_ptr, acc, mask=mask_q)

def context_attention_fwd(q, k, v, o, causal=False, sm_scale=None):
    BLOCK_D = q.shape[-1]
    assert BLOCK_D in {16, 32, 64, 128}, "Unsupported head dimension"
    
    seqlen_q, seqlen_k = q.shape[-2], k.shape[-2]
    device = q.device
    
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    BLOCK_M = 128 if torch.cuda.get_device_capability(device)[0] >= 8 else 64
    BLOCK_N = 64
    kv_group_num = q.shape[1] // k.shape[1]
    
    grid = (q.shape[0], q.shape[1], (seqlen_q + BLOCK_M - 1) // BLOCK_M)
    
    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        seqlen_q, seqlen_k, q.shape[-1],
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal, KV_GROUP_NUM=kv_group_num,
        num_warps=4,
        num_stages=4,
    )
