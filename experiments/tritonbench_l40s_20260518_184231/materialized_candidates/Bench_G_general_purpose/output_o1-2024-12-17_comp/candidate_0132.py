import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_heads, n_ctx, d_model, M,
    USE_FP8: tl.constexpr, IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_hb = tl.program_id(1)

    # batch/head extraction
    head_idx = pid_hb % n_heads
    b_idx = pid_hb // n_heads

    # row indices for Q/out
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # column indexing
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # pointer offsets
    q_ptrs = Q_ptr + (b_idx * stride_qb + head_idx * stride_qh) \
                       + (offs_m[:, None] * stride_qm) + (offs_d[None, :] * stride_qd)
    out_ptrs = Out_ptr + (b_idx * stride_ob + head_idx * stride_oh) \
                          + (offs_m[:, None] * stride_om) + (offs_d[None, :] * stride_od)

    # registers
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < d_model), other=0.)
    # accumulate partial sums for the final output
    out_acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # for numerical stability in softmax
    max_scores = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    sum_scores = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over n_ctx in steps of BLOCK_N
    n_block_end = tl.cdiv(n_ctx, BLOCK_N)
    for block_n in range(n_block_end):
        # column indices within the K/V block
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # load K
        k_ptrs = K_ptr + (b_idx * stride_kb + head_idx * stride_kh) \
                         + (offs_n[None, :] * stride_kn) + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd
        if USE_FP8:
            k = tl.load(k_ptrs, mask=(offs_n[None, :] < n_ctx) & (tl.arange(0, BLOCK_DMODEL)[:, None] < d_model), other=0).to(tl.float16)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[None, :] < n_ctx) & (tl.arange(0, BLOCK_DMODEL)[:, None] < d_model), other=0.)

        # compute attention logits = Q * K^T
        # q is [BLOCK_M, d_model], k is [d_model, BLOCK_N]
        # note that q/k have shape transposed in code from above index expansions
        att_logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d_block in range(0, d_model, BLOCK_DMODEL):
            q_slice = q[:, d_block : d_block + BLOCK_DMODEL]
            k_slice = k[d_block : d_block + BLOCK_DMODEL, :]
            partial = tl.dot(q_slice.to(tl.float32), k_slice.to(tl.float32))
            att_logits += partial
        att_logits *= sm_scale

        # causal masking
        if IS_CAUSAL:
            row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            col_offset = block_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
            mask = (col_offset > row_offset)
            att_logits = tl.where(mask, float('-inf'), att_logits)

        # update max_scores
        cur_max = tl.maximum(tl.max(att_logits, 1), max_scores)
        # scaling factor to avoid overflow
        exp_diff = tl.exp(att_logits - cur_max[:, None])
        # update sums
        old_scale = tl.exp(max_scores - cur_max)
        sum_scores = sum_scores * old_scale + tl.sum(exp_diff, 1)
        max_scores = cur_max

        # compute partial softmax
        att_weights = exp_diff / sum_scores[:, None]

        # load V
        v_ptrs = V_ptr + (b_idx * stride_vb + head_idx * stride_vh) \
                         + (offs_n[None, :] * stride_vn) + tl.arange(0, BLOCK_DMODEL)[:, None] * stride_vd
        if USE_FP8:
            v = tl.load(v_ptrs, mask=(offs_n[None, :] < n_ctx) & (tl.arange(0, BLOCK_DMODEL)[:, None] < d_model), other=0).to(tl.float16)
        else:
            v = tl.load(v_ptrs, mask=(offs_n[None, :] < n_ctx) & (tl.arange(0, BLOCK_DMODEL)[:, None] < d_model), other=0.)

        # accumulate partial output
        # att_weights is [BLOCK_M, BLOCK_N], v is [BLOCK_N, d_model] (with transposition from index expansions)
        for d_block in range(0, d_model, BLOCK_DMODEL):
            v_slice = v[tl.arange(d_block, d_block + BLOCK_DMODEL), :]
            w_slice = att_weights[:, :]
            partial_out = tl.dot(w_slice.to(tl.float32), v_slice.to(tl.float32).transpose())
            out_acc[:, d_block : d_block + BLOCK_DMODEL] += partial_out

    # final normalization for numerical stability
    # we have sum_scores and max_scores after all blocks
    # So the final out_acc = out_acc / sum_scores with dimension broadcast
    out_acc = out_acc / sum_scores[:, None]
    # store back
    tl.store(
        out_ptrs,
        out_acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < d_model)
    )

def triton_fa(q, k, v, out, sm_scale, is_causal=False, use_fp8=False):
    # shapes
    batch, n_heads, m_size, d_model = q.shape
    _, _, n_ctx, _ = k.shape

    BLOCK = 64
    BLOCK_DMODEL = 32

    grid_m = triton.cdiv(m_size, BLOCK)
    grid_hb = n_heads * batch

    # choose num_warps
    # approximate heuristic
    num_warps = 4 if n_ctx <= 64 else 8

    _fwd_kernel[grid_m, grid_hb](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_heads, n_ctx, d_model, m_size,
        use_fp8, is_causal,
        BLOCK, BLOCK, BLOCK_DMODEL,
        num_stages=2,
        num_warps=num_warps
    )
    return out
