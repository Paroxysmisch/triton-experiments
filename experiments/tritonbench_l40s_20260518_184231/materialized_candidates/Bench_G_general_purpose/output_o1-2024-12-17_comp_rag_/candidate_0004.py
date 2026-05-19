import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q,
    K,
    V,
    B0,
    sm_scale,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_b0bh, stride_b0rw, stride_b0cl,
    Out,
    stride_obs, stride_oh,
    seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (pid_b * stride_qbs + pid_h * stride_qh) + (offs_m[:, None] * stride_qbs) + offs_d[None, :]
    k_ptrs_base = K + (pid_b * stride_kbs + (pid_h) * stride_kh)
    v_ptrs_base = V + (pid_b * stride_vbs + (pid_h) * stride_vh)
    b0_ptrs_base = B0 + (pid_b * stride_b0bh + pid_h * stride_b0bh)

    # Load query
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len[pid_b], other=0.0)

    # Initialize partial softmax values
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over sequence dimension
    for start_n in range(0, seq_len[pid_b], BLOCK_N):
        cur_offs_n = start_n + offs_n
        # Load key
        k = tl.load(
            k_ptrs_base + (cur_offs_n[None, :] * stride_kbs) + offs_d[:, None],
            mask=cur_offs_n[None, :] < seq_len[pid_b],
            other=0.0,
        )
        # Compute qk
        qk = tl.dot(q, k)
        # Apply scale
        qk *= sm_scale

        # Load bias from B0 (assuming shape matches properly)
        # b0_ptrs: row = offs_m, col = cur_offs_n
        b0_ptrs = b0_ptrs_base + (offs_m[:, None] * stride_b0rw) + (cur_offs_n[None, :] * stride_b0cl)
        bias_val = tl.load(b0_ptrs, mask=(offs_m[:, None] < seq_len[pid_b]) & (cur_offs_n[None, :] < seq_len[pid_b]), other=0.0)
        qk += bias_val

        # Apply causal mask if needed (prevent attending to future positions)
        mask = offs_m[:, None] >= cur_offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        # Softmax update
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = (l_i / l_i_new) * alpha
        acc = acc * acc_scale[:, None]

        # Load value
        v = tl.load(
            v_ptrs_base + (cur_offs_n[:, None] * stride_vbs) + offs_d[None, :],
            mask=cur_offs_n[:, None] < seq_len[pid_b],
            other=0.0,
        )
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

    # Store result
    out_ptrs = Out + (pid_b * stride_obs + pid_h * stride_oh) + (offs_m[:, None] * stride_obs) + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len[pid_b])


def _attention_rel_h_rel_w_kernel_aligned_device(
    Q,
    K,
    V,
    B0,
    Out,
    seq_len,
    sm_scale,
    BLOCK_M=64,
    BLOCK_N=64,
    BLOCK_DMODEL=64,
    num_warps=4,
    num_stages=2,
):
    import torch
    batch = Q.shape[0]
    heads = Q.shape[1]

    # Validate shape compatibility
    assert Q.shape[0] == K.shape[0] == V.shape[0] == B0.shape[0], "Batch size mismatch"
    assert Q.shape[1] == K.shape[1] == V.shape[1] == B0.shape[1], "Head size mismatch"
    assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == BLOCK_DMODEL, "Feature size mismatch"

    grid = (batch, heads, triton.cdiv(int(seq_len.max().item()), BLOCK_M))

    _fwd_kernel_aligned[grid](
        Q, K, V, B0,
        sm_scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        B0.stride(0), B0.stride(2), B0.stride(3),
        Out,
        Out.stride(0), Out.stride(1),
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Out
