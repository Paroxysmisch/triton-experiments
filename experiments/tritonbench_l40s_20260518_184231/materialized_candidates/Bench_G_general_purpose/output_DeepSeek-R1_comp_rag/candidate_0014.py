import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                    start_m, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                    BLOCK_N: tl.constexpr, STAGE: tl.constexpr, offs_m, offs_n,
                    N_CTX: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_offs_n = start_n + offs_n
        k_mask = k_offs_n < N_CTX
        k = tl.load(K_ptrs + start_n * HEAD_DIM, mask=k_mask[None, :], other=0.0)
        k_scale = tl.load(K_scale_ptr + (start_n // BLOCK_N))
        qk = tl.dot(q, k, allow_tf32=False).to(tl.float32) * q_scale * k_scale
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1e6)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs + start_n * HEAD_DIM, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16), allow_tf32=False).to(tl.float32)
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        V_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
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

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptrs = K + qvk_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    Q_scale_ptr = Q_scale + off_hz * (N_CTX // BLOCK_M) + start_m
    K_scale_ptr = K_scale + off_hz * (N_CTX // BLOCK_N)
    O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q_scale = tl.load(Q_scale_ptr)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr,
                                    V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE,
                                    offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < N_CTX)

def forward(q, k, v, q_scale, k_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM == k.shape[-1] == v.shape[-1], "Head dimensions must match"
    Z, H, N_CTX, _ = q.shape
    o = torch.empty_like(q)

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=3,
        num_warps=8,
        num_stages=3
    )
    return o
