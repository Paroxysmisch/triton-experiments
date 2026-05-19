import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr, seeds_ptr, offsets_ptr, output_ptr,
    batch_size, num_tokens,
    BLOCK: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Define program IDs
    batch_idx = tl.program_id(0)
    token_idx = tl.arange(0, BLOCK_N)

    # Compute strides
    scores_stride = num_tokens
    output_stride = 1

    # Load scores for the current batch
    scores = tl.load(scores_ptr + batch_idx * scores_stride + token_idx, mask=token_idx < num_tokens, other=0.0)

    # Initialize cumulative sum and random sample
    cumsum = tl.zeros([BLOCK_N], dtype=tl.float32)
    random_sample = tl.load(seeds_ptr + batch_idx) + tl.load(offsets_ptr + batch_idx)

    # Compute cumulative distribution
    for i in range(0, num_tokens, BLOCK_N):
        block_scores = tl.load(scores_ptr + batch_idx * scores_stride + i + token_idx, mask=token_idx < num_tokens - i, other=0.0)
        cumsum += block_scores
        mask = random_sample < cumsum
        if tl.any(mask):
            # Find the first token where the random sample falls
            selected_token = i + tl.argmin(mask)
            tl.store(output_ptr + batch_idx * output_stride, selected_token)
            return

    # In case of numerical issues, select the last token
    tl.store(output_ptr + batch_idx * output_stride, num_tokens - 1)


import torch

def multinomial_sampling(scores, seeds, offsets):
    # Ensure scores is a 2D tensor
    assert scores.ndim == 2
    batch_size, num_tokens = scores.shape

    # Define block sizes
    BLOCK = 8
    BLOCK_N = 128

    # Prepare output tensor
    output = torch.empty((batch_size,), dtype=torch.int32, device=scores.device)

    # Launch Triton kernel
    grid = (batch_size,)
    _multinomial_sampling_kernel[grid](
        scores, seeds, offsets, output,
        batch_size, num_tokens,
        BLOCK=BLOCK, BLOCK_N=BLOCK_N
    )

    return output
