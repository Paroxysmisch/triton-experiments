import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch = pid // (N_CTX * N_CTX)
    head = (pid % (N_CTX * N_CTX)) // N_CTX
    start_m = (pid % (N_CTX * N_CTX)) % N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (batch * stride_qz + head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (batch * stride_kz + head * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None])
    v_ptrs = V + (batch * stride_vz + head * stride_vh + offs_n[None, :] * stride_vk + offs_d[:, None])
    out_ptrs = Out + (batch * stride_oz + head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :])

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q_scale = tl.load(Q_scale + batch * stride_qz + head * stride_qh + offs_m[:, None] * stride_qm, mask=offs_m[:, None] < N_CTX, other=1.0)
    k_scale = tl.load(K_scale + batch * stride_kz + head * stride_kh + offs_n[None, :] * stride_kn, mask=offs_n[None, :] < N_CTX, other=1.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn, mask=start_n + offs_n[None, :] < N_CTX, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= q_scale * k_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + start_n * stride_vk, mask=start_n + offs_n[:, None] < N_CTX, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def context_attention_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    N_CTX,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_DMODEL=64
):
    if triton.config['device']['cuda_capability'][0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    grid = (N_CTX * N_CTX * Q.shape[0],)
    num_warps = 4 if BLOCK_DMODEL <= 64 else 8

    _attn_fwd[grid](
        Q,
        K,
        V,
        Q_scale,
        K_scale,
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        N_CTX,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1
    )
