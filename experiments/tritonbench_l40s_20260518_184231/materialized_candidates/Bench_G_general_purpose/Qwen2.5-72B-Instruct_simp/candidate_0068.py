import triton
import triton.language as tl
import torch

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr,  # Pointer to the scores tensor
    seeds_ptr,   # Pointer to the seeds tensor
    offsets_ptr, # Pointer to the offsets tensor
    outputs_ptr, # Pointer to the output tensor
    batch_size,  # Batch size
    num_tokens,  # Number of tokens
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)  # Get the program ID
    batch_idx = pid  # Each program handles one batch

    # Load the seed and offset for this batch
    seed = tl.load(seeds_ptr + batch_idx)
    offset = tl.load(offsets_ptr + batch_idx)

    # Initialize the random number generator
    rng_state = seed + offset

    # Initialize the cumulative sum and max score
    cumulative_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    max_score = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)

    # Iterate over tokens in the block
    for token_idx in range(0, num_tokens, BLOCK_SIZE):
        # Load scores for this block
        scores = tl.load(scores_ptr + batch_idx * num_tokens + token_idx, mask=token_idx + tl.arange(0, BLOCK_SIZE) < num_tokens, other=-float('inf'))

        # Compute the max score in this block
        max_score_block = tl.max(scores, axis=0)
        max_score = tl.where(max_score > max_score_block, max_score, max_score_block)

        # Subtract the max score to avoid overflow
        scores -= max_score

        # Compute the exponential of the scores
        exp_scores = tl.exp(scores)

        # Compute the cumulative sum of the exponential scores
        cumulative_sum += exp_scores

        # Normalize the cumulative sum
        cumulative_sum /= cumulative_sum[-1]

        # Generate a random number
        rng_state = tl.xorshift32(rng_state)
        random_num = tl.rand(rng_state)

        # Find the index where the random number falls in the cumulative sum
        for i in range(BLOCK_SIZE):
            if random_num < cumulative_sum[i]:
                tl.store(outputs_ptr + batch_idx, token_idx + i)
                break

def multinomial_sampling(scores, seeds, offsets):
    batch_size, num_tokens = scores.shape
    outputs = torch.empty((batch_size,), dtype=torch.int32, device=scores.device)

    # Define the grid and block sizes
    grid = (batch_size,)
    block = (128,)

    # Launch the kernel
    _multinomial_sampling_kernel[grid, block](
        scores, seeds, offsets, outputs, batch_size, num_tokens, BLOCK_SIZE=128
    )

    return outputs
