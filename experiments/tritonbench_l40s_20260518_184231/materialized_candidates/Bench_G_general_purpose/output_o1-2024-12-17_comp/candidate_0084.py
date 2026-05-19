import triton
import triton.language as tl


# Constants for block sizes
BLOCK_M = 64
BLOCK_N = 64
BLOCK_DMODEL = 64


@triton.jit
def _fwd_kernel_aligned(
    Q_PTR, K_PTR, V_PTR, B0_PTR, REL_H_W_PTR, OUT_PTR,
    stride_qb, stride_qh, stride_qd,  # Strides for Q
    stride_kb, stride_kh, stride_kd,  # Strides for K
    stride_vb, stride_vh, stride_vd,  # Strides for V
    stride_ob, stride_oh, stride_od,  # Strides for Out
    stride_b0,                        # Stride for bias
    stride_rhw,                       # Stride for rel_h_w
    batch_size, num_heads, seq_len, d_model,
    sm_scale, BIAS_LAST_SIZE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    head_idx = pid_b % num_heads
    batch_idx = pid_b // num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Pointers to the relevant block for Q
    q_ptrs = Q_PTR + batch_idx * stride_qb + head_idx * stride_qh + (offs_m[:, None] * stride_qd) + offs_d[None, :]  # [BLOCK_M, BLOCK_DMODEL]

    # Initialize accumulators for the final output
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # For the softmax denominator
    max_logits = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    sum_exp_logits = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over K / V blocks
    num_iter = seq_len // BLOCK_N
    for i in range(num_iter):
        koffs_n = i * BLOCK_N + offs_n
        # Pointers to K and V blocks
        k_ptrs = K_PTR + batch_idx * stride_kb + head_idx * stride_kh + (koffs_n[None, :] * stride_kd) + offs_d[:, None]  # [BLOCK_DMODEL, BLOCK_N]
        v_ptrs = V_PTR + batch_idx * stride_vb + head_idx * stride_vh + (koffs_n[None, :] * stride_vd) + offs_d[:, None]  # [BLOCK_DMODEL, BLOCK_N]

        # Load Q and K
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0)
        k = tl.load(k_ptrs, mask=(offs_d[:, None] < d_model) & (koffs_n[None, :] < seq_len), other=0.0)
        # Compute Q*K
        logit = tl.dot(q, k)

        # Scale
        logit = logit * sm_scale

        # Add bias from B0 (and potentially rel_h_w if desired)
        if BIAS_LAST_SIZE > 0:
            b0_ptrs = B0_PTR + (batch_idx * num_heads + head_idx) * stride_b0 + offs_m[:, None] * BIAS_LAST_SIZE + koffs_n[None, :]
            bias_vals = tl.load(b0_ptrs, mask=(offs_m[:, None] < seq_len) & (koffs_n[None, :] < seq_len), other=0.0)
            logit += bias_vals

        # Optionally add rel_h_w
        rel_ptrs = REL_H_W_PTR + head_idx * stride_rhw + offs_m[:, None] * seq_len + koffs_n[None, :]
        rel_vals = tl.load(rel_ptrs, mask=(offs_m[:, None] < seq_len) & (koffs_n[None, :] < seq_len), other=0.0)
        logit += rel_vals

        # Update max logits
        curr_max = tl.maximum(tl.max(logit, 1), max_logits)
        exp_logit = tl.exp2(logit - curr_max[:, None])
        old_exp = tl.exp2(max_logits - curr_max)
        sum_exp_logits = old_exp * sum_exp_logits + tl.sum(exp_logit, 1)
        max_logits = curr_max

        # Normalize partial scores and accumulate
        denom = 1.0 / sum_exp_logits
        exp_logit = exp_logit * denom[:, None]

        # Load V
        v = tl.load(v_ptrs, mask=(offs_d[:, None] < d_model) & (koffs_n[None, :] < seq_len), other=0.0)

        # Weighted sum
        acc_scale = exp_logit.to(tl.float32)
        acc += tl.dot(acc_scale, tl.trans(v))

    # Final normalize with updated max_logits
    # No final normalization needed for the output if we've aggregated. The LRC trick has been applied in each iteration.

    # Store to Out
    out_ptrs = OUT_PTR + batch_idx * stride_ob + head_idx * stride_oh + (offs_m[:, None] * stride_od) + offs_d[None, :]
    mask_m = offs_m[:, None] < seq_len
    mask_d = offs_d[None, :] < d_model
    tl.store(out_ptrs, acc, mask=(mask_m & mask_d))


def _attention_rel_h_rel_w_kernel_aligned_device(
    q, k, v, b0, rel_h_w, out, sm_scale
):
    assert q.dtype == k.dtype == v.dtype, "Q, K, V must have the same dtype."
    assert len(q.shape) == 4, "Q must be [batch, head, seq_len, d_model]."
    assert len(k.shape) == 4, "K must be [batch, head, seq_len, d_model]."
    assert len(v.shape) == 4, "V must be [batch, head, seq_len, d_model]."
    assert len(rel_h_w.shape) == 3, "rel_h_w must be [head, seq_len, seq_len]."
    assert q.shape == k.shape == v.shape, "Q, K, V must have the same shape."
    assert b0.shape[0] == q.shape[0] * q.shape[1], "Bias first dim must match batch*head."
    assert b0.shape[1] == q.shape[2], "Bias second dim must match seq_len."
    assert b0.shape[2] == q.shape[2], "Bias last dim must match seq_len."
    BATCH, HEAD, SEQ_LEN, D_MODEL = q.shape
    BIAS_LAST_SIZE = b0.shape[2]

    grid = ( (SEQ_LEN + BLOCK_M - 1) // BLOCK_M, BATCH * HEAD )

    _fwd_kernel_aligned[grid](
        q, k, v, b0, rel_h_w, out,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3),
        v.stride(0), v.stride(1), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        b0.stride(0),
        rel_h_w.stride(0),
        BATCH, HEAD, SEQ_LEN, D_MODEL,
        sm_scale, BIAS_LAST_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
