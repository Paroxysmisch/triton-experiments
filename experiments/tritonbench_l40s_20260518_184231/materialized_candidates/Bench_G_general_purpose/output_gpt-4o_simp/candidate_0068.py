import triton
import triton.language as tl
import torch

@triton.jit
def _multinomial_sampling_kernel(scores_ptr, seeds_ptr, offsets_ptr, outputs_ptr,
                                 batch_size, num_tokens, BLOCK_SIZE: tl.constexpr):
    # Define the block index and block size
    batch_idx = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE

    # Load seeds and offsets
    seed = tl.load(seeds_ptr + batch_idx)
    offset = tl.load(offsets_ptr + batch_idx)

    # Initialize cumulative sum and random threshold
    cumulative_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    rand_threshold = tl.rand(seed, offset) * tl.sum(tl.load(scores_ptr + batch_idx * num_tokens, mask=tl.arange(0, BLOCK_SIZE) < num_tokens))

    # Iterate over the tokens within the block
    for i in range(BLOCK_SIZE):
        token_idx = block_start + i
        if token_idx < num_tokens:
            score = tl.load(scores_ptr + batch_idx * num_tokens + token_idx)
            cumulative_sum[i] = score + (cumulative_sum[i - 1] if i > 0 else 0)

            # Check if cumulative sum exceeds the random threshold
            if cumulative_sum[i] > rand_threshold:
                tl.store(outputs_ptr + batch_idx, token_idx)
                break

def multinomial_sampling(scores, seeds, offsets):
    # Get dimensions
    batch_size, num_tokens = scores.shape

    # Allocate output tensor
    outputs = torch.empty((batch_size,), dtype=torch.int32, device=scores.device)

    # Launch the Triton kernel
    BLOCK_SIZE = 128  # Define an appropriate block size
    grid = (batch_size, (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _multinomial_sampling_kernel[grid](
        scores_ptr=scores,
        seeds_ptr=seeds,
        offsets_ptr=offsets,
        outputs_ptr=outputs,
        batch_size=batch_size,
        num_tokens=num_tokens,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return outputs

# Example usage
batch_size = 1024
num_tokens = 10000
scores = torch.rand((batch_size, num_tokens), device='cuda')
seeds = torch.randint(0, 2**31, (batch_size,), dtype=torch.int32, device='cuda')
offsets = torch.randint(0, 2**31, (batch_size,), dtype=torch.int32, device='cuda')

sampled_indices = multinomial_sampling(scores, seeds, offsets)
print(sampled_indices)
