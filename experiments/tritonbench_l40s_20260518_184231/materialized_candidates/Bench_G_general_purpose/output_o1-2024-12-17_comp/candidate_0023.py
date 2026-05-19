import triton
import triton.language as tl


# --------------------------------------------------------------------------------
# rotary_embedding_kernel
# --------------------------------------------------------------------------------
@triton.jit
def rotary_embedding_kernel(
    Q_PTR, K_PTR,
    COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR,
    stride_q_batch, stride_q_head, stride_q_dmodel,
    stride_k_batch, stride_k_head, stride_k_dmodel,
    stride_cos_batch, stride_cos_head, stride_cos_offset,
    stride_sin_batch, stride_sin_head, stride_sin_offset,
    B, H, N,  # batch size, number of heads, sequence length
    HEAD_DIM,  # dimension per head
    **meta
):
    pid = tl.program_id(0)
    # Each program handles a batch-head-sequence block
    batch_id = pid // (H * N)
    hid_nid = pid % (H * N)
    head_id = hid_nid // N
    seq_id = hid_nid % N

    # Indices for loads
    offset_q0 = batch_id * stride_q_batch + head_id * stride_q_head + seq_id * stride_q_dmodel
    offset_k0 = batch_id * stride_k_batch + head_id * stride_k_head + seq_id * stride_k_dmodel

    offset_cos = batch_id * stride_cos_batch + head_id * stride_cos_head + seq_id * stride_cos_offset
    offset_sin = batch_id * stride_sin_batch + head_id * stride_sin_head + seq_id * stride_sin_offset

    # We assume HEAD_DIM is even, so we split HEAD_DIM into two halves
    # For the rotation, we load data in pairs: (q0, q1), (k0, k1)
    # We'll iterate in blocks
    BLOCK_SIZE = meta["BLOCK_SIZE"]
    for d in range(0, HEAD_DIM, BLOCK_SIZE):
        d_range = tl.arange(0, BLOCK_SIZE)
        mask = d + d_range < HEAD_DIM
        # Load q
        q_offset = offset_q0 + (d + d_range)
        q_val = tl.load(Q_PTR + q_offset, mask=mask, other=0.0)
        # Load cos / sin
        cos_val = tl.load(COS_PTR + offset_cos + (d + d_range), mask=mask, other=0.0)
        sin_val = tl.load(SIN_PTR + offset_sin + (d + d_range), mask=mask, other=0.0)
        # For half-dim rotations, we do pairing
        half = HEAD_DIM // 2
        # q0 = first half
        # q1 = second half
        # Indices for second half
        second_offset = q_offset + half
        second_mask = mask & (d + d_range + half < HEAD_DIM)
        q_val_half = tl.load(Q_PTR + second_offset, mask=second_mask, other=0.0)

        # out_q0 = q0 * cos - q1 * sin
        # out_q1 = q0 * sin + q1 * cos
        out_q0 = q_val * cos_val - q_val_half * sin_val
        out_q1 = q_val * sin_val + q_val_half * cos_val

        # Store output
        tl.store(Q_OUT_PTR + q_offset, out_q0, mask=mask)
        tl.store(Q_OUT_PTR + second_offset, out_q1, mask=second_mask)

        # If K_PTR is provided, do the same for K
        if K_PTR is not None:
            k_val = tl.load(K_PTR + offset_k0 + (d + d_range), mask=mask, other=0.0)
            k_val_half = tl.load(K_PTR + offset_k0 + (d + d_range + half), mask=second_mask, other=0.0)

            out_k0 = k_val * cos_val - k_val_half * sin_val
            out_k1 = k_val * sin_val + k_val_half * cos_val

            tl.store(K_OUT_PTR + offset_k0 + (d + d_range), out_k0, mask=mask)
            tl.store(K_OUT_PTR + offset_k0 + (d + d_range + half), out_k1, mask=second_mask)


# --------------------------------------------------------------------------------
# fused_rotary_embedding_kernel_v2
# (Handles the case with k_cache)
# --------------------------------------------------------------------------------
@triton.jit
def fused_rotary_embedding_kernel_v2(
    Q_PTR, K_PTR, K_CACHE_PTR,
    COS_PTR, SIN_PTR,
    BLOCK_TABLES_PTR,
    KV_LENGTHS_PTR,
    Q_OUT_PTR, K_OUT_PTR,
    stride_q_batch, stride_q_head, stride_q_dmodel,
    stride_k_batch, stride_k_head, stride_k_dmodel,
    stride_cos_batch, stride_cos_head, stride_cos_offset,
    stride_sin_batch, stride_sin_head, stride_sin_offset,
    B, H, N,  # batch size, number of heads, sequence length
    HEAD_DIM,  # dimension per head
    **meta
):
    pid = tl.program_id(0)
    # Each program handles a batch-head-sequence block
    batch_id = pid // (H * N)
    hid_nid = pid % (H * N)
    head_id = hid_nid // N
    seq_id = hid_nid % N

    offset_q0 = batch_id * stride_q_batch + head_id * stride_q_head + seq_id * stride_q_dmodel
    offset_k0 = batch_id * stride_k_batch + head_id * stride_k_head + seq_id * stride_k_dmodel
    offset_cos = batch_id * stride_cos_batch + head_id * stride_cos_head + seq_id * stride_cos_offset
    offset_sin = batch_id * stride_sin_batch + head_id * stride_sin_head + seq_id * stride_sin_offset

    # For block tables
    # block_offset helps identify where in k_cache to store k
    # kv_lengths is the past sequence length
    block_idx = tl.load(BLOCK_TABLES_PTR + pid)
    kv_length = tl.load(KV_LENGTHS_PTR + batch_id)
    cache_offset = block_idx * HEAD_DIM + head_id * kv_length * HEAD_DIM

    BLOCK_SIZE = meta["BLOCK_SIZE"]
    for d in range(0, HEAD_DIM, BLOCK_SIZE):
        d_range = tl.arange(0, BLOCK_SIZE)
        mask = d + d_range < HEAD_DIM
        q_offset = offset_q0 + (d + d_range)
        q_val = tl.load(Q_PTR + q_offset, mask=mask, other=0.0)

        cos_val = tl.load(COS_PTR + offset_cos + (d + d_range), mask=mask, other=0.0)
        sin_val = tl.load(SIN_PTR + offset_sin + (d + d_range), mask=mask, other=0.0)

        half = HEAD_DIM // 2
        second_offset = q_offset + half
        second_mask = mask & (d + d_range + half < HEAD_DIM)
        q_val_half = tl.load(Q_PTR + second_offset, mask=second_mask, other=0.0)

        out_q0 = q_val * cos_val - q_val_half * sin_val
        out_q1 = q_val * sin_val + q_val_half * cos_val

        tl.store(Q_OUT_PTR + q_offset, out_q0, mask=mask)
        tl.store(Q_OUT_PTR + second_offset, out_q1, mask=second_mask)

        if K_PTR is not None:
            k_val = tl.load(K_PTR + offset_k0 + (d + d_range), mask=mask, other=0.0)
            k_val_half = tl.load(K_PTR + offset_k0 + (d + d_range + half), mask=second_mask, other=0.0)

            out_k0 = k_val * cos_val - k_val_half * sin_val
            out_k1 = k_val * sin_val + k_val_half * cos_val

            k_out_offset = offset_k0 + (d + d_range)
            tl.store(K_OUT_PTR + k_out_offset, out_k0, mask=mask)
            tl.store(K_OUT_PTR + k_out_offset + half, out_k1, mask=second_mask)

            # store into k_cache
            cache_block_offset = cache_offset + seq_id * HEAD_DIM
            cache_offset_0 = cache_block_offset + d + d_range
            tl.store(K_CACHE_PTR + cache_offset_0, k_val, mask=mask)
            tl.store(K_CACHE_PTR + cache_offset_0 + half, k_val_half, mask=second_mask)


# --------------------------------------------------------------------------------
# Python wrapper functions
# --------------------------------------------------------------------------------
def rotary_embedding(q, k, cos, sin, q_out, k_out, B, H, N, HEAD_DIM, num_warps=4, block_size=64, stream=None):
    grid = (B * H * N,)
    rotary_embedding_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0) if k is not None else 0,
        k.stride(1) if k is not None else 0,
        k.stride(2) if k is not None else 0,
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        B, H, N, HEAD_DIM,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
        stream=stream
    )


def fused_rotary_embedding_v2(q, k, k_cache, cos, sin,
                              block_tables, kv_lengths,
                              q_out, k_out,
                              B, H, N, HEAD_DIM, num_warps=4, block_size=64, stream=None):
    grid = (B * H * N,)
    fused_rotary_embedding_kernel_v2[grid](
        q, k, k_cache, cos, sin,
        block_tables, kv_lengths,
        q_out, k_out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0) if k is not None else 0,
        k.stride(1) if k is not None else 0,
        k.stride(2) if k is not None else 0,
        cos.stride(0), cos.stride(1), cos.stride(2),
        sin.stride(0), sin.stride(1), sin.stride(2),
        B, H, N, HEAD_DIM,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
        stream=stream
    )
