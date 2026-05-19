import torch
import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr,
    cum_scores_ptr,
    indices_ptr,
    batch_size,
    num_tokens,
    stride_batch_in_scores,
    stride_batch_in_cum_scores,
    stride_batch_in_indices,
    stride_token_in_scores,
    stride_token_in_cum_scores,
    seed,
    offset,
    BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(0)
    # Initialize sampling seeds and offsets
    init_sampling_seeds(seed, offset, batch_idx, batch_size)
    # Generate random samples
    samples = tl.randint(0, num_tokens, (BLOCK,))
    # Compute cumulative scores block-wise
    cum_scores = tl.zeros((BLOCK,), dtype=tl.float32)
    scores_block_ptr = (
        scores_ptr
        + batch_idx * stride_batch_in_scores
        + tl.arange(0, BLOCK) * stride_token_in_scores
    )
    scores = tl.load(scores_block_ptr, mask=tl.arange(0, BLOCK) < num_tokens, other=0.0)
    cum_scores += scores
    # Determine token indices for samples
    token_indices = tl.where(samples < cum_scores, samples, 0)
    # Store the result
    cum_scores_block_ptr = (
        cum_scores_ptr
        + batch_idx * stride_batch_in_cum_scores
        + tl.arange(0, BLOCK) * stride_token_in_cum_scores
    )
    tl.store(cum_scores_block_ptr, cum_scores, mask=tl.arange(0, BLOCK) < num_tokens)
    indices_block_ptr = (
        indices_ptr
        + batch_idx * stride_batch_in_indices
        + tl.arange(0, BLOCK) * 1
    )
    tl.store(indices_block_ptr, token_indices, mask=tl.arange(0, BLOCK) < num_tokens)

def multinomial_sampling(scores, cum_scores, indices, seed=None, offset=None):
    # Set block sizes
    BLOCK = 8
    BLOCK_N = 128
    # Create grid for parallel execution
    grid = (scores.shape[0],)
    # Compute necessary strides
    stride_batch_in_scores = scores.stride(0)
    stride_token_in_scores = scores.stride(1)
    stride_batch_in_cum_scores = cum_scores.stride(0)
    stride_token_in_cum_scores = cum_scores.stride(1)
    stride_batch_in_indices = indices.stride(0)
    # Prepare seed and offset
    if seed is None:
        seed = torch.empty(1, dtype=torch.int64)
        torch.random.manual_seed(torch.randint(0, 10000000, seed.shape))
        seed = seed.item()
    if offset is None:
        offset = torch.zeros(1, dtype=torch.int64)
    offset = offset.item()
    # Call the Triton kernel
    _multinomial_sampling_kernel[grid](
        scores,
        cum_scores,
        indices,
        scores.shape[0],
        scores.shape[1],
        stride_batch_in_scores,
        stride_batch_in_cum_scores,
        stride_batch_in_indices,
        stride_token_in_scores,
        stride_token_in_cum_scores,
        seed,
        offset,
        BLOCK=BLOCK,
        BLOCK_N=BLOCK_N,
    )
    return seed, offset + 1
