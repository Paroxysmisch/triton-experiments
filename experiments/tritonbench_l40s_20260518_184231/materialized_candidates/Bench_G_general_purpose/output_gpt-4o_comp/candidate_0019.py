import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def _fwd_kernel_token_softmax(
    Logits, B_Start_Loc, B_Seqlen, Prob_Out,
    BLOCK_SIZE: tl.constexpr,
    ):
    # Determine the batch and head index
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Calculate the start location and sequence length for the current batch
    start_loc = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)

    # Determine the start index for this batch and head
    start_idx = start_loc + head_id * seq_len

    # Load the logits for this sequence
    logits_ptr = Logits + start_idx
    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < seq_len, other=-float('inf'))

    # Compute the maximum for numerical stability
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits

    # Exponentiate the logits
    exp_logits = tl.exp(logits)

    # Compute the sum of exponentials
    sum_exp_logits = tl.sum(exp_logits, axis=0)

    # Normalize to get probabilities
    probs = exp_logits / sum_exp_logits

    # Store the results in the output tensor
    prob_out_ptr = Prob_Out + start_idx
    tl.store(prob_out_ptr + tl.arange(0, BLOCK_SIZE), probs, mask=tl.arange(0, BLOCK_SIZE) < seq_len)

# Define the wrapper function
@torch.no_grad()
def token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out, max_input_len):
    # Determine the block size and number of warps
    BLOCK_SIZE = min(128, max_input_len)
    num_warps = 4 if BLOCK_SIZE > 64 else 2

    # Calculate grid dimensions
    num_batches = b_start_loc.shape[0]
    num_heads = logits.shape[1] // max_input_len
    grid = (num_batches, num_heads)

    # Launch the Triton kernel
    _fwd_kernel_token_softmax[grid](
        logits, b_start_loc, b_seqlen, prob_out,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
