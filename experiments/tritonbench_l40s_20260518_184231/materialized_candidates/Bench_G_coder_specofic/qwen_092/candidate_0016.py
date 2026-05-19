triton
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, sm_scale, B_Start_Loc, B_Seqlen,
    BLOCK_M: int, BLOCK_DMODEL: int, BLOCK_N: int,
    SEQUENCE_LEN: int, HEADS: int, BATCH_SIZE: int,
    SEQUENCE_BLOCKS: int, WARPS_PER_BLOCK: int
):
    # Get the indices of the current block
    batch_idx = triton.program_id(0)
    head_idx = triton.program_id(1)
    block_idx = triton.program_id(2)

    # Compute the starting and ending indices for the current block
    start_idx = block_idx * BLOCK_N
    end_idx = min(start_idx + BLOCK_N, SEQUENCE_LEN)

    # Initialize the output tensor for the current block
    out = triton.zeros((BLOCK_M, BLOCK_N), dtype=triton.float32)

    # Iterate over the K and V blocks
    for k_block in range(SEQUENCE_BLOCKS):
        k_start_idx = k_block * BLOCK_N
        k_end_idx = min(k_start_idx + BLOCK_N, SEQUENCE_LEN)

        # Load Q, K, and V blocks
        q = Q[batch_idx, head_idx, start_idx:end_idx, :]
        k = K[batch_idx, head_idx, k_start_idx:k_end_idx, :]
        v = V[batch_idx, head_idx, k_start_idx:k_end_idx, :]

        # Compute attention scores (QK^T) with sliding window constraint
        qk = triton.dot(q, k.T) * sm_scale

        # Apply softmax to the attention scores
        qk_max = triton.max(qk, axis=1, keepdim=True)
        qk_exp = triton.exp(qk - qk_max)
        qk_sum = triton.sum(qk_exp, axis=1, keepdim=True)
        attention_probs = qk_exp / qk_sum

        # Accumulate the output
        out += triton.dot(attention_probs, v)

    # Store the output to the final output tensor
    Out[batch_idx, head_idx, start_idx:end_idx, :] = out
