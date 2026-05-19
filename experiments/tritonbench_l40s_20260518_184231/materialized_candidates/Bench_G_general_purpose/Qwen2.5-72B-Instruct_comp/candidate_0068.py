import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr,  # Pointer to the scores tensor
    indices_ptr,  # Pointer to the output indices tensor
    batch_size,  # Number of batches
    num_tokens,  # Number of tokens per batch
    seed,  # Random seed
    BLOCK: tl.constexpr,  # Block size for batches
    BLOCK_N: tl.constexpr,  # Block size for tokens
):
    # Compute the block and thread indices
    pid = tl.program_id(axis=0)
    num_blocks = (batch_size + BLOCK - 1) // BLOCK
    block_start = pid * BLOCK
    block_end = min(block_start + BLOCK, batch_size)

    # Compute the stride for the scores tensor
    stride_scores = num_tokens

    # Initialize the random number generator
    rng = tl.rand(seed + pid)

    # Iterate over the batches in this block
    for batch_idx in range(block_start, block_end):
        # Initialize the cumulative sum
        cumulative_sum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Initialize the indices
        indices = tl.zeros((BLOCK_N,), dtype=tl.int32)

        # Iterate over the tokens in this batch
        for token_idx in range(0, num_tokens, BLOCK_N):
            # Load the scores for this block of tokens
            scores = tl.load(scores_ptr + batch_idx * stride_scores + token_idx, mask=token_idx + tl.arange(0, BLOCK_N) < num_tokens, other=0.0)

            # Compute the cumulative sum for this block of tokens
            cumulative_sum += scores

            # Generate random samples for this block of tokens
            random_samples = rng.rand(BLOCK_N)

            # Determine the token indices where each random sample falls within the cumulative probability range
            for i in range(BLOCK_N):
                if random_samples[i] < cumulative_sum[i]:
                    indices[i] = token_idx + i

        # Store the sampled indices
        tl.store(indices_ptr + batch_idx, indices, mask=indices < num_tokens)

import torch

def multinomial_sampling(scores, seed):
    # Convert the input tensor to a Triton-compatible format
    scores = scores.contiguous()
    batch_size, num_tokens = scores.shape

    # Allocate the output tensor for the sampled indices
    indices = torch.empty((batch_size,), dtype=torch.int32, device=scores.device)

    # Define the block and grid sizes
    BLOCK = 8
    BLOCK_N = 128
    grid = (batch_size + BLOCK - 1) // BLOCK

    # Launch the kernel
    _multinomial_sampling_kernel[grid, (BLOCK,)](
        scores_ptr=scores,
        indices_ptr=indices,
        batch_size=batch_size,
        num_tokens=num_tokens,
        seed=seed,
        BLOCK=BLOCK,
        BLOCK_N=BLOCK_N,
    )

    return indices
