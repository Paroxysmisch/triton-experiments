import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr, output_ptr, batch_size, num_tokens, num_samples_per_token,
    BLOCK: tl.constexpr, BLOCK_N: tl.constexpr
):
    # scores_ptr: (batch_size, num_tokens)
    # output_ptr: (batch_size, num_tokens, num_samples_per_token)
    pid = tl.program_id(axis=0)
    # The block of the program will sample all the tokens in a batch.
    batch_offset = pid * BLOCK
    # The index of the current token to sample.
    token_idx = tl.arange(0, BLOCK_N)
    # The sampled token indices are stored in the output tensor in the following format:
    # (batch_idx_1, token_idx_1, sampled_token_idx_1, sampled_token_idx_2, ...)
    # (batch_idx_1, token_idx_2, sampled_token_idx_1, sampled_token_idx_2, ...)
    # (batch_idx_2, token_idx_1, sampled_token_idx_1, sampled_token_idx_2, ...)
    # (batch_idx_2, token_idx_2, sampled_token_idx_1, sampled_token_idx_2, ...)
    output_ptr += batch_offset * num_tokens * num_samples_per_token + token_idx * num_samples_per_token
    scores_ptr += batch_offset * num_tokens
    # 1. Sample seeds for all the tokens in the current block.
    # The random function generates the same sequence for the same seeds.
    # This ensures that the same program instance will reproduce the same samples.
    seed = batch_offset * BLOCK_N + token_idx
    # 2. Compute the cumulative scores.
    # Iterate over the scores column-wise.
    cum_score = tl.zeros([BLOCK], dtype=tl.float32)
    # The stride represents how much we need to increase the pointer to advance 1 row.
    stride = num_tokens - BLOCK_N
    for _ in range(BLOCK_N):
        # Load the scores for the current column.
        # The scores are loaded into a register, which is faster to access than global memory.
        # However, this also means that we cannot sample more tokens than there are registers available.
        # This is why we can only sample up to 8192 tokens at a time.
        orig_scores = tl.load(scores_ptr + token_idx, mask=token_idx < num_tokens - BLOCK_N, other=0.0)
        # Advance the scores pointer to the next column.
        scores_ptr += stride
        # We also need to advance the output pointer to the next column.
        output_ptr += BLOCK_N
        # Compute the new cumulative scores.
        cum_score += orig_scores
        # Sample a random number for each token.
        # This random number will be compared with the cumulative scores.
        rand = tl.rand(seed)
        # We want to find the first index where rand < cum_score, which is equivalent to
        # finding the index where cum_score crosses rand for the first time.
        # This can be done with a binary search.
        # The lower bound of the search range is 0.
        lower = tl.zeros([BLOCK], dtype=tl.int32)
        # The upper bound of the search range is the number of tokens.
        upper = tl.full([BLOCK], num_tokens - BLOCK_N, dtype=tl.int32)
        # We use a while loop to perform the binary search.
        # The while loop will be unrolled by Triton, so we don't actually need a loop.
        # This is equivalent to a for loop with a fixed number of iterations.
        while lower < upper:
            mid = (lower + upper) // 2
            mask = mid < num_tokens - BLOCK_N
            if rand < tl.load(cum_score + mid, mask=mask):
                upper = mid
            else:
                lower = mid + 1
        # The result is stored in lower.
        # We write the result to the output tensor.
        # The output tensor is of shape (batch_size, num_tokens, num_samples_per_token).
        # We can use the following formula to compute the output pointer:
        # output_ptr + batch_offset * num_tokens * num_samples_per_token + token_idx * num_samples_per_token + (num_tokens - BLOCK_N - lower)
        # This can be further simplified to:
        tl.store(output_ptr + (num_tokens - BLOCK_N - lower), token_idx, mask=(num_tokens - BLOCK_N - lower) < num_samples_per_token)

def multinomial_sampling(scores: Tensor, num_samples_per_token: int):
    assert scores.ndim == 2
    batch_size, num_tokens = scores.shape
    output = torch.empty((batch_size, num_tokens, num_samples_per_token), dtype=torch.int32, device=scores.device)
    # The block size is the number of tokens that each program instance will sample.
    # The block size must be divisible by the number of warps.
    # 8192 is the maximum block size that can be used on A100 with CUDA 11.6.
    # 4096 is the maximum block size that can be used on A100 with CUDA 11.4.
    # 2048 can be used on A100 with CUDA 11.4 if the number of warps is 2 or 4.
    # 1024 can be used on A100 with CUDA 11.4 if the number of warps is 8.
    BLOCK = 8
    # The number of blocks is the number of program instances that will be launched.
    # The grid size must be divisible by the block size.
    # The grid size is limited by the maximum number of active blocks per multiprocessor.
    # On A100, the maximum number of active blocks per multiprocessor is 20.
    # Therefore, the maximum grid size is 20 * num_multiprocessors.
    # On A100, num_multiprocessors is 80.
    # So the maximum grid size is 20 * 80 = 1600.
    # However, we also need to consider the maximum number of threads per block.
    # The maximum number of threads per block is 1024.
    # Therefore, the maximum grid size is also limited by the maximum number of threads.
    # The maximum grid size is 1024 * 1600 = 1638400.
    # However, we have set the block size to 8192, so the maximum grid size is limited to 1638400 / 8192 = 20.
    # The grid size is automatically calculated by Triton based on the BLOCK size.
    # We can set it to 1 to launch a single block that will handle the entire tensor.
    grid = (triton.cdiv(batch_size, BLOCK),)
    num_warps = 4
    _multinomial_sampling_kernel[grid](
        scores, output, batch_size, num_tokens, num_samples_per_token,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return output
