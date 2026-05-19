import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, sm_scale, OUT,
    stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh,
    stride_obs, stride_oh, stride_b0,
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
    b0_ptrs = B0 + (cur_batch * stride_b0 + offs_m[:, None] * stride_b0 + offs_n[None, :])

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q.shape[0], other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < K.shape[0], other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < V.shape[0], other=0.0)
    b0 = tl.load(b0_ptrs, mask=offs_m[:, None] < B0.shape[0], other=0.0)

    qk = tl.dot(q, k) * sm_scale + b0
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    acc += tl.dot(p, v)

    out_ptrs = OUT + (cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None] * stride_obs + offs_d[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < OUT.shape[0])


def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, OUT, sm_scale):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = Q.shape[-1]

    grid = (Q.shape[0], Q.shape[1], triton.cdiv(Q.shape[2], BLOCK_M))
    num_warps = 4 if BLOCK_DMODEL <= 64 else 8

    _fwd_kernel_aligned[grid](
        Q, K, V, B0, sm_scale, OUT,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1),
        V.stride(0), V.stride(1), OUT.stride(0), OUT.stride(1),
        B0.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps, num_stages=1
    )
