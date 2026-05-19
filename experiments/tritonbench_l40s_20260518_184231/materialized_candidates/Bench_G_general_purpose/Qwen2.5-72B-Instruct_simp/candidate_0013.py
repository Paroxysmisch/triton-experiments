import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen, B_req_idx,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    nheads, seq_len, kv_group_num, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    head_id = tl.program_id(axis=2)

    # Compute the start and end indices for the sequence
    start_loc = tl.load(B_Start_Loc + bid)
    seqlen = tl.load(B_Seqlen + bid)
    req_idx = tl.load(B_req_idx + bid)

    # Compute the block indices
    block_m = pid * BLOCK_M
    block_n = head_id * (seq_len // kv_group_num) + (bid % kv_group_num) * BLOCK_N

    # Compute the offsets for Q, K, and V
    q_offset = bid * stride_qb + head_id * stride_qh
    k_offset = bid * stride_kb + (head_id // kv_group_num) * stride_kh
    v_offset = bid * stride_vb + (head_id // kv_group_num) * stride_vh

    # Initialize the output block
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # Compute the attention scores
    for n in range(0, seqlen, BLOCK_N):
        k_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(seq_len, BLOCK_DMODEL),
            strides=(stride_kd, stride_kd),
            offsets=(n, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(seq_len, BLOCK_DMODEL),
            strides=(stride_qd, stride_qd),
            offsets=(block_m, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        q = tl.load(q_block_ptr)
        k = tl.load(k_block_ptr)
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale

        # Apply the softmax function
        qk = tl.softmax(qk, axis=1)

        # Compute the weighted values
        v_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(seq_len, BLOCK_DMODEL),
            strides=(stride_vd, stride_vd),
            offsets=(n, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        v = tl.load(v_block_ptr)
        acc += tl.dot(qk, v, allow_tf32=True)

    # Store the output
    out_block_ptr = tl.make_block_ptr(
        base=Out + bid * stride_ob + head_id * stride_oh,
        shape=(seq_len, BLOCK_DMODEL),
        strides=(stride_od, stride_od),
        offsets=(block_m, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(out_block_ptr, acc)

import triton
import torch

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, B_req_idx, nheads, seq_len, kv_group_num, sm_scale):
    # Get the shapes and strides
    B, H, T, D = Q.shape
    assert K.shape == (B, H // kv_group_num, T, D)
    assert V.shape == (B, H // kv_group_num, T, D)
    assert B_Start_Loc.shape == (B,)
    assert B_Seqlen.shape == (B,)
    assert B_req_idx.shape == (B,)

    # Allocate the output tensor
    Out = torch.empty_like(Q)

    # Configure the grid and block dimensions
    grid = (T // 128, B, H)
    block = (128, 128, 1)

    # Launch the kernel
    _fwd_kernel[grid, block](
        Q, K, V, Out,
        B_Start_Loc, B_Seqlen, B_req_idx,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        nheads, seq_len, kv_group_num, sm_scale,
        BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=D
    )

    return Out
