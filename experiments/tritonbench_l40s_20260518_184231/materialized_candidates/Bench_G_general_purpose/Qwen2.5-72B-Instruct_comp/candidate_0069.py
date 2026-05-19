import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, Out,
    block_count, block_offset, column_count, column_index,
    seqlens, sm_scale, qk_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_km,
    stride_vb, stride_vh, stride_vm,
    stride_ob, stride_oh, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    MAX_BLOCK_COUNT: tl.constexpr, MAX_COLUMN_COUNT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // (MAX_BLOCK_COUNT * MAX_COLUMN_COUNT)
    block_id = (pid % (MAX_BLOCK_COUNT * MAX_COLUMN_COUNT)) // MAX_COLUMN_COUNT
    column_id = (pid % (MAX_BLOCK_COUNT * MAX_COLUMN_COUNT)) % MAX_COLUMN_COUNT

    if block_id >= block_count[bid] or column_id >= column_count[bid]:
        return

    block_start = block_offset[bid, block_id]
    column_start = column_index[bid, column_id]

    q_offset = bid * stride_qb + block_start * BLOCK_M
    k_offset = bid * stride_kb + column_start * BLOCK_N
    v_offset = bid * stride_vb + column_start * BLOCK_N
    o_offset = bid * stride_ob + block_start * BLOCK_M

    q_ptr = Q + q_offset
    k_ptr = K + k_offset
    v_ptr = V + v_offset
    o_ptr = Out + o_offset

    q = tl.load(q_ptr, mask=tl.arange(0, BLOCK_M)[:, None] < seqlens[bid], other=0.0)
    k = tl.load(k_ptr, mask=tl.arange(0, BLOCK_N) < seqlens[bid], other=0.0)
    v = tl.load(v_ptr, mask=tl.arange(0, BLOCK_N) < seqlens[bid], other=0.0)

    q = q * qk_scale
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n in range(0, BLOCK_N, BLOCK_D):
        k_block = k[:, n:n + BLOCK_D]
        v_block = v[:, n:n + BLOCK_D]
        qk = tl.dot(q, k_block, trans_b=True)
        qk = tl.where(tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_D)[None, :], qk, float('-inf'))
        qk = tl.softmax(qk * sm_scale, axis=1)
        acc += tl.dot(qk, v_block)

    tl.store(o_ptr, acc, mask=tl.arange(0, BLOCK_M)[:, None] < seqlens[bid])

import torch

def _triton_mixed_sparse_attention(Q, K, V, block_count, block_offset, column_count, column_index, seqlens, sm_scale, qk_scale):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64
    MAX_BLOCK_COUNT = 1024
    MAX_COLUMN_COUNT = 1024

    B, H, M = Q.shape
    _, _, N = K.shape
    _, _, D = V.shape

    Out = torch.empty((B, H, M), device=Q.device, dtype=Q.dtype)

    grid = (B * MAX_BLOCK_COUNT * MAX_COLUMN_COUNT,)

    _triton_mixed_sparse_attn_fwd_kernel[grid](
        Q, K, V, Out,
        block_count, block_offset, column_count, column_index,
        seqlens, sm_scale, qk_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_D, MAX_BLOCK_COUNT, MAX_COLUMN_COUNT
    )

    return Out
