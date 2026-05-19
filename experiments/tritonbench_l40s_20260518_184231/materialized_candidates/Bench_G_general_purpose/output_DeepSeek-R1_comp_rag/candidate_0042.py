import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, K_scale, V_scale, Out,
    softmax_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    qvk_offset = off_z * stride_qz + off_h * stride_qh
    K_scale_offset = off_hz * (N_CTX + BLOCK_N - 1) // BLOCK_N
    V_scale_offset = off_hz * (N_CTX + BLOCK_N - 1) // BLOCK_N

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptrs = Q + qvk_offset + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    K_ptrs = K + qvk_offset + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
    V_ptrs = V + qvk_offset + (offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn)
    O_ptrs = Out + qvk_offset + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, start_m * BLOCK_M if BLOCK_M > 0 else N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[None, :] < N_CTX, other=0)
        k = k.to(tl.int8)
        k_scale = tl.load(K_scale + K_scale_offset + (start_n // BLOCK_N))
        k_dequant = k.to(tl.float16) * k_scale

        qk = tl.dot(q, k_dequant, out_dtype=tl.float32) * softmax_scale

        mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk = tl.where(mask, qk, float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs + start_n * stride_vk, mask=(start_n + offs_n)[:, None] < N_CTX, other=0)
        v = v.to(tl.int8)
        v_scale = tl.load(V_scale + V_scale_offset + (start_n // BLOCK_N))
        v_dequant = v.to(tl.float16) * v_scale

        acc += tl.dot(p.to(tl.float16), v_dequant, out_dtype=tl.float32)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX)

def context_attention_fwd_ppl_int8kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                    k_scale: torch.Tensor, v_scale: torch.Tensor,
                                    log_factor: float = 1.0):
    assert q.dtype == torch.float16
    assert k.dtype == torch.int8 and v.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16

    Z, H, N_CTX, HEAD_DIM = q.shape
    BLOCK_M = 128
    BLOCK_N = 64 if HEAD_DIM <= 64 else 128
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    inv_sqrt_dim = (HEAD_DIM ** -0.5)
    softmax_scale = inv_sqrt_dim * log_factor

    out = torch.empty_like(q)
    
    _fwd_kernel_int8kv[grid](
        q, k, v, k_scale, v_scale, out,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=4
    )
    return out
