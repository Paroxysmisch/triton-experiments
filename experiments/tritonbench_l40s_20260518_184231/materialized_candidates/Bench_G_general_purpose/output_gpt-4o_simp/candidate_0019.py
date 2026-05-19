import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 128  # This can be tuned based on your hardware and problem size

@triton.jit
def _fwd_kernel_token_softmax(Logits, Prob_Out, stride_logits_m, stride_logits_h, stride_logits_t, stride_prob_m, stride_prob_h, stride_prob_t, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the current block
    pid = tl.program_id(0)
    
    # Compute the batch, head, and token indices
    batch_idx = pid // (stride_logits_h * stride_logits_t)
    head_idx = (pid // stride_logits_t) % stride_logits_h
    token_idx = pid % stride_logits_t

    # Compute the offset in the logits and output tensors
    logits_offset = batch_idx * stride_logits_m + head_idx * stride_logits_h + token_idx * stride_logits_t
    prob_offset = batch_idx * stride_prob_m + head_idx * stride_prob_h + token_idx * stride_prob_t

    # Load the logits for the current token sequence
    logits = tl.load(Logits + logits_offset + tl.arange(0, BLOCK_SIZE))

    # Compute the max for numerical stability
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits

    # Compute the exponentials
    exp_logits = tl.exp(logits)

    # Compute the sum of exponentials
    sum_exp_logits = tl.sum(exp_logits, axis=0)

    # Compute the probabilities
    probabilities = exp_logits / sum_exp_logits

    # Store the result
    tl.store(Prob_Out + prob_offset + tl.arange(0, BLOCK_SIZE), probabilities)

def token_softmax_fwd(logits, prob_out, batch_size, num_heads, seq_length):
    # Calculate the strides for the logits and output tensors
    stride_logits_m = num_heads * seq_length
    stride_logits_h = seq_length
    stride_logits_t = 1

    stride_prob_m = num_heads * seq_length
    stride_prob_h = seq_length
    stride_prob_t = 1

    # Calculate the number of blocks needed
    num_blocks = batch_size * num_heads * seq_length // BLOCK_SIZE

    # Launch the kernel
    _fwd_kernel_token_softmax[(num_blocks,)](
        logits, prob_out,
        stride_logits_m, stride_logits_h, stride_logits_t,
        stride_prob_m, stride_prob_h, stride_prob_t,
        BLOCK_SIZE=BLOCK_SIZE
    )
