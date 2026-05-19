import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q, K, V, Out,
    q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + pid_z * stride_kz + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + pid_z * stride_vz + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    q = tl.load(q_ptrs, mask=offs_m[:, None] < stride_qm, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < stride_kn, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < stride_vk, other=0.0)

    qk = tl.dot(q, k) * q_scale * k_scale
    m_i = tl.max(qk, 1)
    p = tl.exp(qk - m_i[:, None])
    l_i = tl.sum(p, 1)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for i in range(0, BLOCK_N):
        acc += p[:, i:i+1] * v[i:i+1, :]

    l_i_new = l_i
    m_i_new = m_i

    out_ptrs = Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc / l_i_new[:, None], mask=offs_m[:, None] < stride_qm)

@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_z = tl.program_id(2)
    for pid_m in range(0, N_CTX // BLOCK_M):
        for pid_n in range(0, N_CTX // BLOCK_N):
            _attn_fwd_inner[pid_m, pid_n, pid_z](
                Q, K, V, Out,
                q_scale, k_scale,
                stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                stride_vz, stride_vh, stride_vk, stride_vn,
                stride_oz, stride_oh, stride_om, stride_on,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL
            )

def context_attention_fwd(Q, K, V, Out, q_scale, k_scale, N_CTX):
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = Q.shape[-1]

    grid = (triton.cdiv(N_CTX, BLOCK_M), triton.cdiv(N_CTX, BLOCK_N), Q.shape[0])

    _attn_fwd[grid](
        Q, K, V, Out,
        q_scale, k_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        N_CTX,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL
    )
