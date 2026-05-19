import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(Q, K, V, Out, stride_qz, stride_qh, stride_qm, stride_qk,
                       stride_kz, stride_kh, stride_kn, stride_kk,
                       stride_vz, stride_vh, stride_vk, stride_vn,
                       stride_oz, stride_oh, stride_om, stride_on,
                       scale_factor, Z, H, N_CTX,
                       BLOCK_DMODEL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    O_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    
    q = tl.load(Q_ptrs)
    k = tl.load(K_ptrs)
    v = tl.load(V_ptrs)
    
    qk = tl.dot(q, k)
    qk = qk * scale_factor
    
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal_mask, qk, float('-inf'))
    
    attn_weights = tl.softmax(qk, axis=-1)
    
    out = tl.dot(attn_weights, v)
    
    tl.store(O_ptrs, out)

def context_attention_fwd_ppl_int8kv(Q, K, V, scale_factor):
    BLOCK_DMODEL = 64
    BLOCK_M = 128
    BLOCK_N = 64
    Z, H, N_CTX, D_MODEL = Q.shape
    
    Out = torch.empty_like(Q)
    
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    
    _fwd_kernel_int8kv[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        scale_factor, Z, H, N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )
    
    return Out
