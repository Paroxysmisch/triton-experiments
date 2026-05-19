import triton
import triton.language as tl

# Define the kernel
@triton.jit
def _fwd_kernel(Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale, BLOCK: tl.constexpr):
    # Get the program ID for parallel execution
    pid = tl.program_id(axis=0)

    # Calculate the batch index and head index
    batch_idx = pid // B_Seqlen.shape[0]
    head_idx = pid % B_Seqlen.shape[0]

    # Calculate start location and sequence length for this batch
    start_loc = B_Start_Loc[batch_idx]
    seqlen = B_Seqlen[batch_idx]

    # Load the query for this block
    q = tl.load(Q + pid * BLOCK)

    # Initialize accumulators for attention scores and outputs
    acc_scores = tl.zeros([BLOCK], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK, V.shape[-1]], dtype=tl.float32)

    # Iterate over blocks of keys and values
    for i in range(0, seqlen, BLOCK):
        # Load key and value blocks
        k = tl.load(K + (start_loc + i) * BLOCK)
        v = tl.load(V + (start_loc + i) * BLOCK * V.shape[-1])

        # Compute dot-product qk and apply scaling
        qk = tl.dot(q, k) * sm_scale

        # Apply attention mask (e.g., causal mask for autoregressive models)
        mask = tl.arange(0, BLOCK) < (seqlen - i)
        qk = tl.where(mask, qk, float('-inf'))

        # Compute softmax scores using numerically stable technique
        max_qk = tl.max(qk, axis=0)
        exp_qk = tl.exp(qk - max_qk)
        sum_exp_qk = tl.sum(exp_qk, axis=0)

        # Update accumulators
        acc_scores += sum_exp_qk
        acc_out += tl.dot(exp_qk, v)

    # Normalize the accumulated outputs by the sum of scores
    acc_out /= acc_scores

    # Write the result to the output tensor
    tl.store(Out + pid * BLOCK * V.shape[-1], acc_out)

# Wrapper function to launch the kernel
def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale, BLOCK=128):
    # Calculate grid size based on batch and head dimensions
    grid = (Q.shape[0] * Q.shape[1],)
    
    # Launch the Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale,
        BLOCK=BLOCK,
        num_warps=4  # Choose number of warps based on performance tuning
    )
