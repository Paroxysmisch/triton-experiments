import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q, K, V, Out,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs, stride_oh,
    Q_scale, K_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (cur_batch * stride_qbs + cur_head * stride_qh + offs_m[:, None] * stride_qbs + offs_d[None, :])
    k_ptrs = K + (cur_batch * stride_kbs + cur_head * stride_kh + offs_n[None, :] * stride_kbs + offs_d[:, None])
    v_ptrs = V + (cur_batch * stride_vbs + cur_head * stride_vh + offs_n[:, None] * stride_vbs + offs_d[None, :])

    q = tl.load(q_ptrs)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, BLOCK_N, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kbs)
        v = tl.load(v_ptrs + start_n * stride_vbs)

        qk = tl.dot(q * Q_scale, k * K_scale)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        acc += tl.dot(p, v)
        l_i += l_ij

    out_ptrs = Out + (cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None] * stride_obs + offs_d[None, :])
    tl.store(out_ptrs, acc / l_i[:, None])

def attention_fwd(Q, K, V, Out, Q_scale, K_scale):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = Q.shape[-1]

    grid = (Q.shape[0], Q.shape[1], triton.cdiv(Q.shape[2], BLOCK_M))
    num_warps = 4

    _attn_fwd_inner[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), Out.stride(0), Out.stride(1),
        Q_scale, K_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
