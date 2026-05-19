import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seqlens_ptr,
    block_offset_ptr, block_count_ptr,
    column_index_ptr, column_count_ptr,
    qk_scale, sm_scale,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    Z, H, N_CTX, block_size, num_col_blocks,
    **meta
):
    # Program IDs for batch-head dimension
    bid = tl.program_id(0)
    # Each bid corresponds to one attention "head" within a batch
    z = bid // H
    h = bid % H

    # Offsets for Q, K, V, Out based on z, h
    Q_off = z * stride_qz + h * stride_qh
    K_off = z * stride_kz + h * stride_kh
    V_off = z * stride_vz + h * stride_vh
    O_off = z * stride_oz + h * stride_oh

    # We iterate over each block in row dimension
    row_blocks = block_count_ptr[bid]
    base_block_offset = block_offset_ptr[bid]

    # Precompute sequence range
    offset_row = seqlens_ptr[z]
    offset_next = seqlens_ptr[z + 1]
    sequence_len = offset_next - offset_row

    # Each row block processes a block_size chunk of Q
    for rb in range(row_blocks):
        row_id = base_block_offset + rb
        row_start = row_id * block_size

        # Load Q sub-block
        q_offset = (Q_off + row_start * stride_qm)
        Q_block = tl.load(Q_ptr + q_offset + tl.arange(0, block_size)[:, None] * stride_qm + tl.arange(0, block_size)[None, :])

        # Apply qk_scale
        Q_block = Q_block * qk_scale

        # Clamp row positions by sequence length
        causal_row_mask = (row_start + tl.arange(0, block_size)) < sequence_len

        # Retrieve column blocks for this row block
        col_blocks = column_count_ptr[row_id]
        col_base_idx = column_index_ptr[row_id * num_col_blocks : row_id * num_col_blocks + col_blocks]

        # Initialize partial output
        out_acc = tl.zeros([block_size, block_size], dtype=tl.float32)

        # Iterate over column blocks
        for cb in range(col_blocks):
            col_id = col_base_idx[cb]
            col_start = col_id * block_size
            # Load K sub-block
            k_offset = (K_off + col_start * stride_kn)
            K_block = tl.load(K_ptr + k_offset + tl.arange(0, block_size)[:, None] * stride_kn + tl.arange(0, block_size)[None, :])
            # Compute dot-product
            attn_scores = tl.dot(Q_block, tl.trans(K_block))
            # Apply causal mask
            causal_col_mask = (col_start + tl.arange(0, block_size)) <= (row_start + tl.arange(0, block_size)[:, None])
            mask = causal_row_mask[:, None] & causal_col_mask
            attn_scores = tl.where(mask, attn_scores, float('-inf'))
            # Scale by sm_scale
            attn_scores = attn_scores * sm_scale
            # Softmax approximation
            max_scores = tl.max(attn_scores, 1)
            attn_scores = attn_scores - max_scores[:, None]
            exp_scores = tl.exp(attn_scores)
            denom = tl.sum(exp_scores, 1)
            # Load V sub-block
            v_offset = (V_off + col_start * stride_vn)
            V_block = tl.load(V_ptr + v_offset + tl.arange(0, block_size)[:, None] * stride_vn + tl.arange(0, block_size)[None, :])
            # Weighted sum
            weight = exp_scores / denom[:, None]
            out_acc += tl.dot(weight, V_block)

        # Write partial results to Out
        out_offset = (O_off + row_start * stride_om)
        tl.store(Out_ptr + out_offset + tl.arange(0, block_size)[:, None] * stride_om + tl.arange(0, block_size)[None, :], out_acc)


def _triton_mixed_sparse_attention(
    Q, K, V, Out,
    seqlens, block_offset, block_count,
    column_index, column_count,
    qk_scale, sm_scale,
    block_size, num_col_blocks
):
    import math
    Z, H, N_CTX, d = Q.shape
    # Strides
    stride_qz = Q.stride(0)
    stride_qh = Q.stride(1)
    stride_qm = Q.stride(2)
    stride_kz = K.stride(0)
    stride_kh = K.stride(1)
    stride_kn = K.stride(2)
    stride_vz = V.stride(0)
    stride_vh = V.stride(1)
    stride_vn = V.stride(2)
    stride_oz = Out.stride(0)
    stride_oh = Out.stride(1)
    stride_om = Out.stride(2)

    grid = (Z * H,)
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        Q, K, V, Out,
        seqlens,
        block_offset, block_count,
        column_index, column_count,
        qk_scale, sm_scale,
        stride_qz, stride_qh, stride_qm,
        stride_kz, stride_kh, stride_kn,
        stride_vz, stride_vh, stride_vn,
        stride_oz, stride_oh, stride_om,
        Z, H, N_CTX, block_size, num_col_blocks
    )
