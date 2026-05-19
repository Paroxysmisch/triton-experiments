import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd

_BLOCK_M = 128
_BLOCK_N = 128
_BLOCK_DMODEL = 512
assert (
    _BLOCK_DMODEL >= _BLOCK_N
), "Error: model dimension is smaller than the chunk size"

@triton.jit(do_not_specialize=["sm_scale", "maxq_value"])
def _fwd_kernel(
    Q,
    K,
    V,
    KVGlobalPtr,
    sm_scale,
    maxq_value,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_b_q,
    stride_b_k,
    stride_b_v,
    stride_b_start_loc,
    stride_b_seqlen,
    stride_head_k,
    stride_head_v,
    stride_head_C,
    stride_head_Q,
    stride_head_B_start_loc_1,
    stride_head_B_seqlen_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    num_block_m: tl.constexpr,
    num_block_n: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_v = tl.program_id(1)
    offs_b = tl.program_id(2)
    offs_head = tl.program_id(3)
    cur_batch = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cur_head = offs_head * BLOCK_N + tl.arange(0, BLOCK_N)
    cur_kv_head = offs_head * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)
    cur_batch_in_all_start_loc = cur_batch + B_Start_Loc
    block_n_num = num_block_n
    cur_seqlen = tl.load(B_Seqlen + cur_batch + stride_b_seqlen)
    block_start_seqlen = tl.load(B_Seqlen + cur_batch + stride_b_start_loc)
    past_seq_mask = tl.where(block_start_seqlen <= cur_seqlen, 0, -1e20)
    Q_ptrs = (
        Q
        + cur_batch_in_all_start_loc * stride_b_q
        + cur_head * stride_head_Q
        + cur_kv_head
    )
    sliding_window_size = cur_seqlen - block_start_seqlen
    end_m = tl.minimum(num_block_m, (((cur_seqlen - 1) // BLOCK_M) + 1))

    for start_n in range(0, end_m):
        start_n = tl.multiple_of(start_n, 1)
        KV_ptrs = (
            KVGlobalPtr
            + cur_kv_head * stride_head_k
            + cur_head.to(tl.int64) * stride_head_V
            + (offs_v * (BLOCK_N) + start_n * BLOCK_N + tl.arange(0, BLOCK_N))
        )
        Q_trans = tl.load(Q_ptrs)
        Q_trans = tl.where(cur_seqlen > block_start_seqlen, Q_trans, 0.0)
        Q_trans_tile = tl.tile(
            Q_trans,
            BLOCK_M,
        )
        QK_max = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        for j_block_n in range(0, block_n_num):
            start_n_j_new = start_n * BLOCK_N
            K_t = tl.load(
                KV_ptrs,
                mask=(start_n_j_new < sliding_window_size),
                other=0.0,
            )
            QK = tl.dot(Q_trans_tile, K_t, allow_tf32=False)
            QK *= sm_scale
            QK_max_new = tl.max(QK, axis=1)
            QK_max_new = tl.where(
                cur_seqlen > (block_start_seqlen + start_n_j_new),
                QK_max_new,
                -float("inf"),
            )
            QK += QK_max_new[:, None] - QK_max
            QK = tl.where(
                cur_seqlen > (block_start_seqlen + start_n_j_new), QK, float("-inf")
            )
            QK = tl.where(QK < maxq_value, QK, maxq_value)
            QK = tl.exp(QK)
            QK += past_seq_mask[:, None]

            V_t = tl.load(
                KV_ptrs + stride_head_k,
                mask=(start_n_j_new < sliding_window_size),
                other=0.0,
            )

            Out_ptrs = (
                Out
                + cur_batch * stride_b_q
                + cur_head * stride_head_C
                + cur_kv_head
                + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) * stride_head_C
                + (offs_v * BLOCK_N + start_n * BLOCK_N + tl.arange(0, BLOCK_N))
            )
            C_t = tl.dot(QK, V_t, allow_tf32=False)
            C_t = tl.where(cur_seqlen > (block_start_seqlen + start_n_j_new), C_t, 0.0)
            tl.store(Out_ptrs, C_t, mask=(cur_seqlen > block_start_seqlen))
            QK = tl.where(cur_seqlen > (block_start_seqlen + start_n_j_new), QK, 0.0)
            QK = QK / tl.sum(QK, axis=1)[:, None]
            tl.store(
                KV_ptrs,
                K_t,
                mask=(start_n_j_new < sliding_window_size) & (offs_b == 0),
            )
            tl.store(
                KV_ptrs + stride_head_k,
                V_t,
                mask=(start_n_j_new < sliding_window_size) & (offs_b == 0),
            )
            past_seq_mask = tl.where(
                cur_seqlen > block_start_seqlen, past_seq_mask, cur_seqlen
            )
            Q_ptrs = Q_ptrs + BLOCK_DMODEL
            KV_ptrs = KV_ptrs + BLOCK_DMODEL * stride_head_k

@custom_fwd
def context_attention_fwd(Q, KV, cu_seqlens, max_seqlen, out, sm_scale,):
    B, head_num = Q.shape[:2]
    KV_shape = KV.shape
    d_model_per_head = KV_shape[2] // 2
    K = KV[:, :, :, : (d_model_per_head)]
    V = KV[:, :, :, (d_model_per_head) :]
    if Q.size(0) * Q.size(1) * Q.size(2)
