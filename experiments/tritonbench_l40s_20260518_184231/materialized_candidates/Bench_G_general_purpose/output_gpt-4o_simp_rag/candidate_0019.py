import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel(Q, K, V, Out, sm_scale, stride_qz, stride_qh, stride_qm, stride_qk, 
                stride_kz, stride_kh, stride_kn, stride_kk, 
                stride_vz, stride_vh, stride_vk, stride_vn, 
                stride_oz, stride_oh, stride_om, stride_on, 
                Z, H, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    
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
    offs_k = tl.arange(0, BLOCK_DMODEL)

    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptrs = K + k_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    O_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

    q = tl.load(Q_ptrs)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_ptrs)
        v = tl.load(V_ptrs)
        qk = tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vk

    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty))

def forward(Q, K, V, sm_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = Q.shape[-1]
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
    Out = torch.empty_like(Q)
    
    grid = (triton.cdiv(Q.shape[2], BLOCK_M), Q.shape[0] * Q.shape[1])
    _fwd_kernel[grid](
        Q, K, V, Out, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Q.shape[0], Q.shape[1], Q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    return Out
