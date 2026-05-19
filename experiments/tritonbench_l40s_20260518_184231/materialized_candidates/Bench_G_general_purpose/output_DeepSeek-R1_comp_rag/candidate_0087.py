import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, 
                    K_ptrs, K_scale_ptr, V_ptrs,  
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    N_CTX: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM)[:, None] < HEAD_DIM)
        k = tl.load(K_ptrs, mask=k_mask, other=0.0)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_mask = (offs_n[:, None] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM)[None, :] < HEAD_DIM)
        v = tl.load(V_ptrs, mask=v_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        V_ptrs += BLOCK_N * HEAD_DIM
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX,  
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    q_scale_offset = off_hz
    k_scale_offset = off_hz
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + qvk_offset + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    Q_scale_ptr = Q_scale + q_scale_offset
    K_ptrs = K + qvk_offset + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + (offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn)
    O_ptrs = Out + qvk_offset + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM), other=0.0)
    q_scale_val = tl.load(Q_scale_ptr)
    k_scale_val = tl.load(K_scale_ptr)
    
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale_val, K_ptrs, K_scale_ptr, V_ptrs,  
                                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    STAGE, offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), 
             mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM))

def forward(q, k, v, q_scale, k_scale):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Head dimensions must match"
    BLOCK_M, BLOCK_N = 128, 64
    HEAD_DIM = q.shape[-1]
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.size(2), BLOCK_M), q.size(0) * q.size(1))
    
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.size(0), q.size(1), q.size(2),
        HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        STAGE=3, num_warps=8, num_stages=3
    )
    return o
