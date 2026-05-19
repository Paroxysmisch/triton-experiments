import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q, K, V,
    K_scale,
    Q_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    Q += off_hz * stride_qh
    K += off_hz * stride_kh
    V += off_hz * stride_vh
    Out += off_hz * stride_oh

    Q += start_m * BLOCK_M * stride_qm
    K += start_m * BLOCK_N * stride_kn
    V += start_m * BLOCK_N * stride_vn
    Out += start_m * BLOCK_M * stride_om

    end_m = min(N_CTX, start_m + BLOCK_M)

    Q_ptrs = Q + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_N)[None, :] * stride_qk
    Q_scale_ptr = Q_scale + off_hz
    K_ptrs = K + tl.arange(0, BLOCK_N)[:, None] * stride_kn + tl.arange(0, BLOCK_M)[None, :] * stride_kk
    K_scale_ptr = K_scale + off_hz
    V_ptrs = V + tl.arange(0, BLOCK_N)[:, None] * stride_vk + tl.arange(0, BLOCK_M)[None, :] * stride_vn
    O_ptrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_N)[None, :] * stride_on

    m_i = tl.full([BLOCK_M, 1], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, end_m, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        Q_curr = tl.load(Q_ptrs + start_n * stride_qk)
        K_curr = tl.load(K_ptrs + start_n * stride_kn)
        V_curr = tl.load(V_ptrs + start_n * stride_vn, mask=mask)
        q_scale_curr = tl.load(Q_scale_ptr)
        k_scale_curr = tl.load(K_scale_ptr)

        qk += q_scale_curr * k_scale_curr * tl.dot(Q_curr, K_curr)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc = acc * tl.exp(m_i - m_ij) + p * V_curr
        m_i = m_ij
        l_i = l_i * tl.exp(m_i - m_ij) + l_ij

    O = acc / l_i
    O = tl.sum(O, 1)
    tl.store(O_ptrs, O)

@torch.no_grad()
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out):
    BLOCK = 128
    Z, H, N_CTX, K = K.shape
    assert Q.shape == (Z, H, N_CTX, K)
    assert Q_scale.shape == (Z, H, N_CTX, 1)
    assert K.shape == Q.shape
    assert K_scale.shape == Q_scale.shape
    assert V.shape == K.shape
    assert Out.shape == Q.shape

    grid = (N_CTX, Z * H)
    _attn_fwd_inner[grid](
        Q, K, V,
        K_scale, Q_scale,
        Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=2,
    )
    return Out
