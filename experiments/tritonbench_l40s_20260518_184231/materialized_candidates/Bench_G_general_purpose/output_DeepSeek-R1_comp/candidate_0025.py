import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    Out, L, M,
    stride_q0, stride_q1, stride_q2, stride_q3,
    stride_k0, stride_k1, stride_k2, stride_k3,
    stride_v0, stride_v1, stride_v2, stride_v3,
    stride_out0, stride_out1, stride_out2, stride_out3,
    stride_m0, stride_m1, stride_m2,
    stride_l0, stride_l1, stride_l2,
    seq_len,
    Lk: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    log2e = 1.44269504  # log2(e)
    batch_head = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    m_start = m_block_idx * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, Lk)
    
    # Initialize pointers to Q
    q_ptr = Q + batch_head * stride_q0 + offs_m[:, None] * stride_q2 + offs_k[None, :] * stride_q3
    mask_q = (offs_m < seq_len)[:, None] & (offs_k < Lk)[None, :]
    q = tl.load(q_ptr, mask=mask_q, other=0.0)
    
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, Lk], dtype=tl.float32)
    
    n_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_block_idx in range(n_blocks):
        n_start = n_block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        
        k_ptr = K + batch_head * stride_k0 + n_offs[:, None] * stride_k2 + offs_k[None, :] * stride_k3
        v_ptr = V + batch_head * stride_v0 + n_offs[:, None] * stride_v2 + offs_k[None, :] * stride_v3
        mask_kv = (n_offs < seq_len)[:, None] & (offs_k < Lk)[None, :]
        k = tl.load(k_ptr, mask=mask_kv, other=0.0)
        v = tl.load(v_ptr, mask=mask_kv, other=0.0)
        
        k_t = tl.trans(k)
        qk = tl.dot(q, k_t) * sm_scale
        
        if IS_CAUSAL:
            causal_mask = (m_start + tl.arange(0, BLOCK_M)[:, None]) >= (n_start + offs_n[None, :])
            qk = tl.where(causal_mask, qk, -float('inf'))
        
        m_curr = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        
        alpha = tl.exp2((qk - m_new[:, None]) * log2e)
        
        l_curr = tl.sum(alpha, axis=1)
        alpha_scale = tl.exp2((m_i - m_new) * log2e)
        l_new = l_i * alpha_scale + l_curr
        
        o_scale = alpha_scale[:, None]
        o_i = o_i * o_scale + tl.dot(alpha, v)
        
        m_i = m_new
        l_i = l_new
    
    o_i = o_i / l_i[:, None]
    
    out_ptr = Out + batch_head * stride_out0 + offs_m[:, None] * stride_out2 + offs_k[None, :] * stride_out3
    tl.store(out_ptr, o_i, mask=mask_q)
    
    m_ptr = M + batch_head * stride_m0 + offs_m * stride_m2
    l_ptr = L + batch_head * stride_l0 + offs_m * stride_l2
    tl.store(m_ptr, m_i, mask=offs_m < seq_len)
    tl.store(l_ptr, l_i, mask=offs_m < seq_len)

def flash_attn_triton(q, k, v, causal=False, sm_scale=None):
    batch_size, num_heads, seq_len, d = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "Input shapes must match"
    
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (d ** 0.5)
    
    o = torch.empty_like(q)
    L = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
    M = torch.empty_like(L)
    
    BLOCK_M = 128
    BLOCK_N = 128 if d >= 64 else 64
    
    grid = (batch_size * num_heads, (seq_len + BLOCK_M - 1) // BLOCK_M, 1)
    
    num_warps = 4 if d <= 64 else 8
    
    _fwd_kernel[grid](
        q, k, v, sm_scale, o, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        seq_len, d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=4
    )
    
    return o
