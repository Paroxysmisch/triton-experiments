import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    q_ptr, k_ptr, att_out_ptr,
    B_Loc_ptr, B_Start_Loc_ptr, B_Seqlen_ptr,
    max_input_len, scale, num_heads, head_dim,
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Compute the start location for the current sequence
    start_loc = tl.load(B_Start_Loc_ptr + batch_idx)
    seq_len = tl.load(B_Seqlen_ptr + batch_idx)

    # Calculate the offset in the q, k, and att_out tensors
    q_offset = (batch_idx * stride_q_batch +
                head_idx * stride_q_head +
                seq_idx * stride_q_seq)
    k_offset = (batch_idx * stride_k_batch +
                head_idx * stride_k_head)
    out_offset = (batch_idx * stride_out_batch +
                  head_idx * stride_out_head +
                  seq_idx * stride_out_seq)

    # Load query vector
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE) * stride_q_dim)

    # Initialize attention output
    att_out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over key vectors
    for k_seq_idx in range(seq_len):
        k = tl.load(k_ptr + k_offset + k_seq_idx * stride_k_seq + tl.arange(0, BLOCK_SIZE) * stride_k_dim)
        # Compute dot product and scale
        dot_product = tl.dot(q, k) * scale
        # Store the result
        tl.store(att_out_ptr + out_offset + k_seq_idx * stride_out_dim, dot_product)

### Wrapper Function: `token_att_fwd`

This function sets up the execution configuration and launches the Triton kernel.
