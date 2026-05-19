import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logics,  # Input logits tensor
    B_Start_Loc,  # Starting indices of each sequence in the batch
    B_Seqlen,  # Length of each sequence
    Prob_Out,  # Output tensor for probabilities
    stride_logic_h,  # Stride for logits in the head dimension
    stride_logic_s,  # Stride for logits in the sequence dimension
    stride_prob_h,  # Stride for probabilities in the head dimension
    stride_prob_s,  # Stride for probabilities in the sequence dimension
    n_head,  # Number of heads
    max_seqlen,  # Maximum sequence length
    BLOCK_SIZE: tl.constexpr  # Block size for processing
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Calculate the starting and ending indices for the current batch and head
    start_loc = tl.load(B_Start_Loc + batch_id)
    end_loc = start_loc + tl.load(B_Seqlen + batch_id)

    # Initialize pointers for the logits and probabilities
    logits_ptr = Logics + batch_id * n_head * max_seqlen + head_id * max_seqlen
    prob_ptr = Prob_Out + batch_id * n_head * max_seqlen + head_id * max_seqlen

    # Load the logits for the current block
    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < end_loc - start_loc, other=-float('inf'))

    # Compute the max value for numerical stability
    max_val = tl.max(logits, axis=0)

    # Subtract the max value from each element
    logits = logits - max_val

    # Compute the exponentials
    exp_logits = tl.exp(logits)

    # Compute the sum of the exponentials
    sum_exp = tl.sum(exp_logits, axis=0)

    # Compute the probabilities
    probs = exp_logits / sum_exp

    # Store the probabilities back to the output tensor
    tl.store(prob_ptr + tl.arange(0, BLOCK_SIZE), probs, mask=tl.arange(0, BLOCK_SIZE) < end_loc - start_loc)

import torch

@torch.no_grad()
def token_softmax_fwd(
    Logics,  # Input logits tensor
    B_Start_Loc,  # Starting indices of each sequence in the batch
    B_Seqlen,  # Length of each sequence
    Prob_Out,  # Output tensor for probabilities
    max_input_len,  # Maximum input length
    n_head  # Number of heads
):
    # Calculate the block size based on the maximum input length
    BLOCK_SIZE = 128  # This can be adjusted based on the maximum input length and GPU capabilities

    # Calculate the number of warps based on the block size
    num_warps = 4 if BLOCK_SIZE >= 512 else 2

    # Launch the kernel
    grid = (Logics.size(0), n_head)  # (batch_size, n_head)
    _fwd_kernel_token_softmax[grid](
        Logics, B_Start_Loc, B_Seqlen, Prob_Out,
        Logics.stride(1), Logics.stride(2),
        Prob_Out.stride(1), Prob_Out.stride(2),
        n_head, max_input_len, BLOCK_SIZE,
        num_warps=num_warps
    )
