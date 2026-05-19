import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q_ptr,                     # [batch_size, num_heads, seqlen_q, d_model] (flattened)
    K_ptr,                     # [batch_size, num_kv_heads, seqlen_k, d_model] (flattened)
    V_ptr,                     # [batch_size, num_kv_heads, seqlen_k, d_model] (flattened)
    layout_csr_row_indices_ptr,  # CSR row offsets (length = num_query_blocks + 1)
    layout_csr_col_indices_ptr,  # CSR col indices (length = total_non_zero_blocks)
    Out_ptr,                   # [batch_size, num_heads, seqlen_q, d_model] (flattened)
    softmax_scale,            # scale factor for softmax (1.0 / sqrt(d_model_block_size))
    batch_size,
    num_heads,
    num_kv_heads,
    q_stride_bs,               # stride for Q in the batch dimension
    q_stride_h,                # stride for Q in the head dimension
    q_stride_seq,              # stride for Q in the sequence dimension
    q_stride_d,
    k_stride_bs,               # stride for K in the batch dimension
    k_stride_h,                # stride for K in the head dimension
    k_stride_seq,
    k_stride_d,
    v_stride_bs,               # stride for V in the batch dimension
    v_stride_h,
    v_stride_seq,
    v_stride_d,
    out_stride_bs,
    out_stride_h,
    out_stride_seq,
    out_stride_d,
    BLOCK_M: tl.constexpr,     # block size in query dimension
    BLOCK_N: tl.constexpr,     # block size in key dimension
    BLOCK_D: tl.constexpr,     # block size in hidden dimension
    NUM_D_BLOCKS: tl.constexpr # how many D blocks to accumulate over
):
    # program_id(0) enumerates (batch_idx * num_heads + head_idx), program_id(1) enumerates query block index
    bh_id = tl.program_id(0)
    row_block_id = tl.program_id(1)

    # Decode batch/head index
    batch_idx = bh_id // num_heads
    head_idx = bh_id % num_heads
    kv_head_idx = head_idx if num_heads == num_kv_heads else (head_idx // (num_heads // num_kv_heads))

    # Row offsets in the CSR
    row_start = tl.load(layout_csr_row_indices_ptr + row_block_id)
    row_end   = tl.load(layout_csr_row_indices_ptr + row_block_id + 1)

    # Query block range in the sequence dimension
    m_range = tl.arange(0, BLOCK_M)
    seq_offset_m = row_block_id * BLOCK_M

    # Allocate local memory for partial softmax
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # For each D-block segment, we will load partial Q; then loop over columns
    # Q has shape [batch_size, num_heads, seqlen_q, d_model], K & V have shape [batch_size, num_kv_heads, seqlen_k, d_model].
    for d_block_id in range(NUM_D_BLOCKS):
        d_offset = d_block_id * BLOCK_D
        # Calculate Q offsets
        offs_m_d = (seq_offset_m + m_range)[:, None] * q_stride_seq + (d_offset + tl.arange(0, BLOCK_D))[None, :]
        q_offset = batch_idx * q_stride_bs + head_idx * q_stride_h + offs_m_d
        q = tl.load(Q_ptr + q_offset, mask=(seq_offset_m + m_range)[:, None] >= 0, other=0.0)

        # Loop over non-zero columns for this row
        col_offset_range = tl.arange(row_start, row_end)
        for c_offset in col_offset_range:
            col_block_id = tl.load(layout_csr_col_indices_ptr + c_offset)

            # Key block offsets in sequence dimension
            n_range = tl.arange(0, BLOCK_N)
            seq_offset_n = col_block_id * BLOCK_N

            # K offset
            offs_n_d = (seq_offset_n + n_range)[None, :] * k_stride_seq + (d_offset + tl.arange(0, BLOCK_D))[:, None]
            k_offset = batch_idx * k_stride_bs + kv_head_idx * k_stride_h + offs_n_d
            k_block = tl.load(K_ptr + k_offset, mask=(seq_offset_n + n_range)[None, :] >= 0, other=0.0)

            # Compute dot product QK
            qk = tl.dot(q, k_block)
            qk = qk * softmax_scale

            # Mask to ensure we only attend to valid positions (right-padding or causal not addressed here if not needed)
            # For a pure block-sparse approach, users typically ensure columns are valid by layout, but if extra check is needed:
            # qk = tl.where((seq_offset_m + m_range)[:, None] <= (seq_offset_n + n_range)[None, :], qk, float('-inf'))

            # Apply stable softmax update
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij

            p_scale = beta / l_i_new
            p = p * p_scale[:, None]

            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]

            # Load V block
            v_offset = batch_idx * v_stride_bs + kv_head_idx * v_stride_h
            v_offset += (seq_offset_n + n_range)[:, None] * v_stride_seq
            v_offset += (d_offset + tl.arange(0, BLOCK_D))[None, :]
            v_block
