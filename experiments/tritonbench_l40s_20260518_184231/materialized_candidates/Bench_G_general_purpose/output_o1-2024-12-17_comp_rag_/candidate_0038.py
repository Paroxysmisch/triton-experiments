import triton
import triton.language as tl
import math
import torch

@triton.jit
def _fwd_kernel_int8kv(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Batch, Heads, Seqlen, Dmodel,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    softmax_scale, log_scale,
    CACHE_SEQLEN,  # can be used for prompt caching logic
    BLOCK_M: tl.constexpr,  # block dimension for the sequence
    BLOCK_N: tl.constexpr,  # block dimension for keys
    BLOCK_DMODEL: tl.constexpr  # block dimension for the model
):
    # program_id determines which block of queries is processed
    pid_m = tl.program_id(0)  # block id along the sequence dimension
    pid_bh = tl.program_id(1)  # combined batch and head dimension
    # batch and head indices
    b_idx = pid_bh // Heads
    h_idx = pid_bh % Heads

    # Starting offset for M dimension (sequence)
    m_start = pid_m * BLOCK_M
    # Ranges for m and n
    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)

    # Compute pointer offsets
    Q_block_ptr = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + m_range[:, None] * stride_qm
    K_block_ptr = K_ptr + b_idx * stride_kb + h_idx * stride_kh
    V_block_ptr = V_ptr + b_idx * stride_vb + h_idx * stride_vh
    Out_block_ptr = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + m_range[:, None] * stride_om

    # Initialize accumulators for partial attention scores and max
    partial_sum = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    m_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)

    # Load Q int8 values and cast to float32
    # shape: [BLOCK_M, Dmodel]
    offs_dk = tl.arange(0, BLOCK_DMODEL)
    q_int8 = tl.load(Q_block_ptr + offs_dk[None, :], mask=(m_range[:, None] < Seqlen) & (offs_dk[None, :] < Dmodel))
    q_fp32 = q_int8.to(tl.float32)

    # We'll iterate over K, V blocks to compute dot-product for the entire sequence dimension
    # n_start can originate from prompt cache offset or be from 0 to Seqlen
    # Here we assume a single pass from 0 to Seqlen with BLOCK_N coverage
    n_offset = 0
    while n_offset < Seqlen:
        # Compute pointer offset for current block of K and V
        k_cur_ptr = K_block_ptr + (n_offset + n_range)[None, :] * stride_kn + offs_dk[:, None]
        v_cur_ptr = V_block_ptr + (n_offset + n_range)[:, None] * stride_vn + offs_dk[None, :]

        # Load K int8 block, cast to float32, shape: [Dmodel, BLOCK_N]
        k_mask = (n_offset + n_range) < Seqlen
        k_int8 = tl.load(k_cur_ptr, mask=(offs_dk[:, None] < Dmodel) & (k_mask[None, :]))
        k_fp32 = k_int8.to(tl.float32)

        # Dot product [BLOCK_M, Dmodel] x [Dmodel, BLOCK_N] => [BLOCK_M, BLOCK_N]
        # Then apply scale
        qk_out = tl.dot(q_fp32, k_fp32) * softmax_scale

        # Apply log-based scale factor if specified
        if log_scale != 0.0:
            qk_out = qk_out * (1.0 / math.log(log_scale))

        # Apply causal mask (autoregressive). For each m in [m_range], mask out n > m
        # But n here is offset by n_offset, so the condition is: n_offset + n_range > m_start + m
        causal_mask = (n_offset + n_range[None, :] > m_range[:, None])
        qk_out = tl.where(causal_mask, -1.0e9, qk_out)

        # Numerically stable softmax update:
        # 1) compute max along last axis
        m_curr = tl.max(qk_out, 1)
        # 2) compare with the running max
        m_new = tl.maximum(m_max, m_curr)
        # 3) exponentiate and sum
        diff_exp = tl.exp(qk_out - m_new[:, None])
        old_scale = tl.exp(m_max - m_new)
        # update l_sum
        l_sum_new = l_sum * old_scale + tl.sum(diff_exp, 1)
        # update partial_sum
        scale_p = old_scale[:, None]
        partial_sum = partial_sum * scale_p

        # Load V int8 block, cast to float32, shape: [BLOCK_N, Dmodel]
        v_int8 = tl.load(v_cur_ptr, mask=(n_offset + n_range[:, None] < Seqlen) & (offs_dk[None, :] < Dmodel))
        v_fp32 = v_int8.to(tl.float32)

        # Weighted sum
        diff_exp = diff_exp.to(tl.float32)
        partial_sum += tl.dot(diff_exp, v_fp32)

        # Update max and sum trackers
        l_sum = l_sum_new
        m_max = m_new

        # Move to the next block of N
        n_offset += BLOCK_N

    # Normalize final result
    denom = 1.0 / l_sum
    out_val = partial_sum * denom[:, None]

    # Write back to Out in float16 or desired precision
    tl.store(Out_block_ptr + offs_dk[None, :], out_val.to(tl.float16),
             mask=(m_range[:, None] < Seqlen) & (offs_dk[None, :] < Dmodel))


def context_attention_fwd_ppl_int8kv(q, k, v, softmax_scale, log_scale=0.0, cache_seqlen=0):
    """
    q, k, v are assumed to be int8 tensors with shape:
      [Batch, Heads, Seqlen, Dmodel]
    Output: out with shape [Batch, Heads, Seqlen, Dmodel], dtype float16

    softmax_scale: can be 1 / sqrt(Dmodel), or any user-defined scale
    log_scale: optional factor to scale via 1/log(...), defaults to 0 (no effect)
    cache_seqlen: if using prompt caching, can offset key/value blocks
    """
    B, H, N, D = q.shape

    out = torch.empty((B, H, N, D), device=q.device, dtype=torch.float16)

    # Block sizes for performance tuning
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = 64  # can be different, chosen for demonstration

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _fwd_kernel_int8kv[grid](
        q, k, v, out,
        B, H, N, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        softmax_scale, log_scale,
        cache_seqlen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=3
    )
    return out
