import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out, 
    Q_scale, K_scale, 
    stride_qz, stride_qh, stride_qm, stride_qk, 
    stride_kz, stride_kh, stride_kn, stride_kk, 
    stride_vz, stride_vh, stride_vk, stride_vn, 
    stride_oz, stride_oh, stride_om, stride_on, 
    Z, H, N_CTX, 
    BLOCK_DMODEL: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    CAUSAL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX)
    q_scale = tl.load(Q_scale_ptr)
    
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (N_CTX - start_n)
        k = tl.load(K_ptrs + start_n * stride_kn, mask=k_mask)
        k_scale = tl.load(K_scale_ptr + start_n // BLOCK_N)
        
        qk = tl.dot(q, k, out_dtype=tl.float32) * q_scale * k_scale
        
        if CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs + start_n * stride_vn, mask=k_mask)
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)
        
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < N_CTX))

def context_attention_fwd_ppl_int8kv(q, k, v, q_scale, k_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    N_CTX = q.shape[2]
    o = torch.empty_like(q, dtype=torch.float16)
    stage = 3
    CAUSAL = True

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    _fwd_kernel_int8kv[grid](
        q, k, v, o,  
        q_scale, k_scale,  
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
        q.shape[0], q.shape[1],  
        N_CTX=N_CTX,  
        BLOCK_DMODEL=BLOCK_DMODEL,  
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,  
        CAUSAL=CAUSAL,  
        num_warps=4,  
        num_stages=4
    )
    return o
