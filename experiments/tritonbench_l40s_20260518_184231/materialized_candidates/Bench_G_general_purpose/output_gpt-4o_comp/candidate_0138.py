import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    softmax_scale: float, num_heads: int, num_kv_heads: int,
    NUM_D_BLOCKS: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Compute batch, head, and block indices
    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads
    kv_head_id = head_id % num_kv_heads

    # Compute row start and end from CSR layout
    row_start = layout_csr_row_indices[pid_m]
    row_end = layout_csr_row_indices[pid_m + 1]

    # Initialize accumulators for the output
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over non-zero blocks in the row
    for idx in range(row_start, row_end):
        col_idx = layout_csr_col_indices[idx]

        # Compute block indices for Q, K, V
        q_offset = pid_m * BLOCK_M
        k_offset = col_idx * BLOCK_N
        v_offset = col_idx * BLOCK_N

        # Load blocks of Q, K, V
        q_block = tl.load(Q + (batch_id * num_heads + head_id) * Q.stride(0) + q_offset * Q.stride(1), mask=True)
        k_block = tl.load(K + (batch_id * num_kv_heads + kv_head_id) * K.stride(0) + k_offset * K.stride(1), mask=True)
        v_block = tl.load(V + (batch_id * num_kv_heads + kv_head_id) * V.stride(0) + v_offset * V.stride(1), mask=True)

        # Compute dot product and apply softmax scaling
        qk = tl.dot(q_block, k_block, trans_b=True) * softmax_scale

        # Apply softmax
        qk_max = tl.max(qk, axis=1, keepdim=True)
        qk_exp = tl.exp(qk - qk_max)
        qk_sum = tl.sum(qk_exp, axis=1, keepdim=True)
        qk_softmax = qk_exp / qk_sum

        # Update the accumulator with the attention-weighted values
        acc += tl.dot(qk_softmax, v_block)

    # Store the accumulated result in the output tensor
    tl.store(out + (batch_id * num_heads + head_id) * out.stride(0) + q_offset * out.stride(1), acc)

def block_sparse_attention(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, BLOCK_M, BLOCK_N, BLOCK_D, softmax_scale, num_heads, num_kv_heads):
    # Determine grid size
    num_query_blocks = layout_csr_row_indices.shape[0] - 1
    batch_size = Q.shape[0] // num_heads
    grid = (num_query_blocks, batch_size * num_heads)

    # Allocate output tensor
    out = torch.empty((Q.shape[0], Q.shape[1], V.shape[2]), device=Q.device, dtype=Q.dtype)

    # Launch the Triton kernel
    block_sparse_attention_kernel[grid](
        Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        softmax_scale=softmax_scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        NUM_D_BLOCKS=(Q.shape[2] + BLOCK_D - 1) // BLOCK_D
    )

    return out
