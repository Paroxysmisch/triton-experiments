import triton
import triton.language as tl

@triton.jit
def parallel_retention_fwd_kernel(
    q, k, v, o, decay_factors, do, output_accum, output_scale,
    block_size_q, block_size_k, block_size_v, block_size_o,
    seq_len, num_heads, head_dim, dtype):
    # Define the block pointers
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(seq_len, block_size_q)

    # Get the block indices
    seq_idx = pid * block_size_q
    seq_end = seq_idx + block_size_q

    # Iterate over the sequence length
    for seq in range(seq_idx, seq_end):
        # Iterate over the attention heads
        for head in range(num_heads):
            # Compute the scaled dot-product
            q_block = q[seq, head, :head_dim]
            k_block = k[seq, head, :head_dim]
            v_block = v[seq, head, :head_dim]
            scaled_dot_product = tl.dot(q_block, k_block) / head_dim

            # Apply decay factor
            decay_factor = decay_factors[head]
            scaled_dot_product *= decay_factor

            # Accumulate the output
            output_accum[seq, head, :head_dim] += scaled_dot_product * v_block

    # Rescale the output
    output_scale[seq_idx:seq_end, :num_heads, :head_dim] = 1.0 / (head_dim * num_heads)
    o[seq_idx:seq_end, :num_heads, :head_dim] = output_accum[seq_idx:seq_end, :num_heads, :head_dim] * output_scale[seq_idx:seq_end, :num_heads, :head_dim]
