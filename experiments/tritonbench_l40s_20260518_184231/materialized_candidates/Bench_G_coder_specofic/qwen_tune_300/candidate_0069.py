import torch
import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, Out,
    block_count, block_offset, column_count, column_index,
    seqlens,
    sm_scale,
    qk_scale,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_block_count_bs, stride_block_count_h,
    stride_column_count_bs, stride_column_count_h,
    stride_column_index_bs, stride_column_index_h,
    stride_seqlens_bs, stride_seqlens_b,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(seqlens + cur_batch * stride_seqlens_bs + 0 * stride_seqlens_b)
    cur_batch_seq_offset = tl.load(seqlens + cur_batch * stride_seqlens_bs + 1 * stride_seqlens_b)

    cur_batch_start_index = BLOCK_SEQ * start_n
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    block_offset_curr = tl.load(
        block_offset + cur_batch * stride_block_count_bs + cur_head * stride_block_count_h,
    )
    column_offset_curr = tl.load(
        column_index + cur_batch * stride_column_index_bs + cur_head * stride_column_index_h,
    )

    pid_n_loop = cur_batch_start_index + tl.arange(0, BLOCK_SEQ)

    accumulator = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_mark in range(0, tl.cdiv(cur_batch_end_index - cur_batch_start_index, BLOCK_N)):
        offs_n = pid_n_loop + start_mark * BLOCK_N
        block_flag_curr = tl.load(
            block_count + block_offset_curr * stride_block_count_bs + offs_n * stride_block_count_h,
        )
        if block_flag_curr == 0:
            continue
        offs_n = tl.arange(0, BLOCK_N) + start_mark * BLOCK_N
        off_k = block_offset_curr * BLOCK_N * stride_kbs + offs_n[None, :] * stride_kh + offs_d[None, :] * stride_kd
        off_v = column_offset_curr * BLOCK_N * stride_vbs + offs_n[:, None] * stride_vh + offs_d[:, None] * stride_vd
        q = tl.load(Q + off_q, mask=offs_n < cur_batch_end_index - cur_batch_start_index, other=0.0)
        k = tl.load(K + off_k, mask=offs_n[None, :] < cur_batch_end_index - cur_batch_start_index, other=0.0)
        v = tl.load(V + off_v, mask=offs_n[:, None] < cur_batch_end_index - cur_batch_start_index, other=0.0)

        q = q.to(tl.float32)
        if qk_scale is not None:
            q *= qk_scale
        q = tl.broadcast_to(q[:, None, :], [BLOCK_SEQ, BLOCK_N, BLOCK_DMODEL])
        qk = tl.sum(q * k, 2)
        qk *= sm_scale

        causal_mask = tl.where(
            (BLOCK_SEQ - (cur_batch_end_index - offs_n))[:, None] >= tl.arange(0, BLOCK_N)[None, :],
            0,
            -10000,
        )
        qk += causal_mask

        block_flag_curr = tl.load(
            block_count + block_offset_curr * stride_block_count_bs + offs_n * stride_block_count_h,
        )
        qk = tl.where(block_flag_curr[:, None] == 0, -10000, qk)

        accumulator += tl.sum(qk * v, axis=1)

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, accumulator)

def _triton_mixed_sparse_attention(
    q, k, v, o,
    block_count, block_offset, column_count, column_index,
    seqlens,
    qk_scale,
    max_seqlen,
    num_warps,
    num_stages,
):
    if qk_scale is not None:
        qk_scale = qk_scale.to(torch.float32)

    sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    BLOCK_SEQ = triton.next_power_of_2(max_seqlen)

    grid = (
        q.shape[0],  # B
        q.shape[1],  # H
        triton.cdiv(q.shape[2], BLOCK_SEQ),  # N / BLOCK_SEQ
    )

    num_warps = num_warps
    num_stages = num_stages

    BLOCK_DMODEL = q.shape[-1]
    BLOCK_N = min(triton.next_power_of_2(k.shape[2]), 64 * 1024)

    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, o,
        block_count, block_offset, column_count, column_index,
        seqlens,
        sm_scale, qk_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        block_count.stride(0), block_count.stride(1),
        column_count.stride(0), column_count.stride(1),
        column_index.stride(0), column_index.stride(1),
        seqlens.stride(0), seqlens.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
