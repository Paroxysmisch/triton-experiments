import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_b0_b, stride_b0_h, stride_b0_sq, stride_b0_sk,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    sm_scale,
    max_seq_len_q, max_seq_len_kv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q block
    q_ptrs = Q + cur_batch * stride_q_b + cur_head * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=offs_m[:, None] < max_seq_len_q, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K, V blocks
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load K block
        k_ptrs = K + cur_batch * stride_k_b + cur_head * stride_k_h + (start_n + offs_n[None, :]) * stride_k_s + offs_d[:, None] * stride_k_d
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < max_seq_len_kv, other=0.0)

        # Compute QK
        qk = tl.dot(q, k)
        qk *= sm_scale

        # Load bias and add
        b_ptrs = B0 + cur_batch * stride_b0_b + cur_head * stride_b0_h + offs_m[:, None] * stride_b0_sq + (start_n + offs_n[None, :]) * stride_b0_sk
        bias = tl.load(b_ptrs, mask=(offs_m[:, None] < max_seq_len_q) & ((start_n + offs_n[None, :]) < max_seq_len_kv), other=0.0)
        qk += bias

        # Causal masking (optional, uncomment if needed)
        # qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float('-inf'))

        # Compute softmax with exp2
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Update accumulator
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        # Load V block
        v_ptrs = V + cur_batch * stride_v_b + cur_head * stride_v_h + (start_n + offs_n[:, None]) * stride_v_s + offs_d[None, :] * stride_v_d
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < max_seq_len_kv, other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i, l_i = m_i_new, l_i_new

    # Store output
    out_ptrs = Out + cur_batch * stride_out_b + cur_head * stride_out_h + offs_m[:, None] * stride_out_s + offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < max_seq_len_q)

def _attention_rel_h_rel_w_kernel_aligned_device(Q, K, V, B0, Out, sm_scale=None):
    assert Q.dtype == K.dtype == V.dtype == B0.dtype, "All inputs must have the same data type"
    assert Q.is_cuda and K.is_cuda and V.is_cuda and B0.is_cuda and Out.is_cuda, "All tensors must be on CUDA device"
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4 and B0.ndim == 4, "Inputs must be 4D tensors"
    assert Q.shape[0] == K.shape[0] == V.shape[0] == B0.shape[0], "Batch size mismatch"
    assert Q.shape[1] == K.shape[1] == V.shape[1] == B0.shape[1], "Head count mismatch"
    assert Q.shape[3] == K.shape[3] == V.shape[3], "Feature dimension mismatch"
    assert B0.shape[2] == Q.shape[2], "B0 q_len must match Q seq_len"
    assert B0.shape[3] == K.shape[2], "B0 k_len must match K seq_len"

    d_model = Q.shape[3]
    if sm_scale is None:
        sm_scale = 1.0 / (d_model ** 0.5)

    BLOCK_M = 128 if triton.runtime.driver.active.get_current_target().capability[0] >= 8 else 64
    BLOCK_N = 64
    BLOCK_DMODEL = d_model

    batch, num_heads, q_len, _ = Q.shape
    max_seq_len_q = q_len
    max_seq_len_kv = K.shape[2]

    grid = (batch, num_heads, triton.cdiv(q_len, BLOCK_M))
    num_warps = 4 if BLOCK_DMODEL <= 64 else 8
    num_stages = 3

    _fwd_kernel_aligned[grid](
        Q, K, V, B0, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B0.stride(0), B0.stride(1), B0.stride(2), B0.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        sm_scale,
        max_seq_len_q, max_seq_len_kv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
