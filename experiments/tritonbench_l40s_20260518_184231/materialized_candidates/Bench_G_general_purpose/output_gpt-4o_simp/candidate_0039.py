import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    batch_size, num_heads, seq_len_q, seq_len_kv, head_dim
):
    # Define the program ID, which identifies the current block of threads
    pid = tl.program_id(0)

    # Calculate the batch and head indices from the program ID
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Define the block size for loading Q, K, and V
    BLOCK_M = 128  # Number of queries processed by each block
    BLOCK_N = 128  # Number of keys processed by each block
    BLOCK_D = 64   # Head dimension

    # Compute the starting index for this block
    q_start = batch_id * stride_qz + head_id * stride_qh
    k_start = batch_id * stride_kz + head_id * stride_kh
    v_start = batch_id * stride_vz + head_id * stride_vh
    o_start = batch_id * stride_oz + head_id * stride_oh

    # Load the query vectors for this block
    q_idx = tl.arange(0, BLOCK_M) + tl.arange(0, BLOCK_D) * stride_qm
    q = tl.load(Q + q_start + q_idx, mask=q_idx < seq_len_q * head_dim, other=0.0)

    # Initialize the output matrix for this block
    out = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over the keys and values in blocks
    for nk in range(0, seq_len_kv, BLOCK_N):
        # Load the key vectors for this block
        k_idx = tl.arange(0, BLOCK_N) + tl.arange(0, BLOCK_D) * stride_kn
        k = tl.load(K + k_start + nk * stride_kn + k_idx, mask=k_idx < seq_len_kv * head_dim, other=0.0)

        # Compute attention scores
        scores = tl.dot(q, k, trans_b=True)

        # Apply softmax to attention scores
        max_score = tl.max(scores, axis=1)
        scores = scores - max_score[:, None]
        exp_scores = tl.exp(scores)
        sum_exp_scores = tl.sum(exp_scores, axis=1)
        softmax_scores = exp_scores / sum_exp_scores[:, None]

        # Load the value vectors for this block
        v_idx = tl.arange(0, BLOCK_N) + tl.arange(0, BLOCK_D) * stride_vn
        v = tl.load(V + v_start + nk * stride_vn + v_idx, mask=v_idx < seq_len_kv * head_dim, other=0.0)

        # Compute the weighted sum of values
        out += tl.dot(softmax_scores, v)

    # Store the output
    out_idx = tl.arange(0, BLOCK_M) + tl.arange(0, BLOCK_D) * stride_om
    tl.store(Out + o_start + out_idx, out, mask=out_idx < seq_len_q * head_dim)

def context_attention_fwd_ppl_int8kv(Q, K, V, batch_size, num_heads, seq_len_q, seq_len_kv, head_dim):
    # Create output tensor
    Out = torch.empty((batch_size, num_heads, seq_len_q, head_dim), dtype=torch.float32, device=Q.device)

    # Calculate strides
    stride_qz, stride_qh, stride_qm = Q.stride()
    stride_kz, stride_kh, stride_kn = K.stride()
    stride_vz, stride_vh, stride_vn = V.stride()
    stride_oz, stride_oh, stride_om = Out.stride()

    # Launch kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel_int8kv[grid](
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm,
        stride_kz, stride_kh, stride_kn,
        stride_vz, stride_vh, stride_vn,
        stride_oz, stride_oh, stride_om,
        batch_size, num_heads, seq_len_q, seq_len_kv, head_dim
    )

    return Out
