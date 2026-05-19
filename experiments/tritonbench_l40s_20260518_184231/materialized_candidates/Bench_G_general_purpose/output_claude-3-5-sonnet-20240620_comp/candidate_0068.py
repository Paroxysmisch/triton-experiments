import triton
import triton.language as tl
import torch

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr,          # pointer to scores [batch_size, num_tokens]
    output_ptr,         # pointer to output indices [batch_size]
    batch_size,         # number of batches
    num_tokens,         # number of tokens per batch
    scores_batch_stride,  # stride for scores batch dimension
    BLOCK: tl.constexpr,  # block size for batch dimension
    BLOCK_N: tl.constexpr,  # block size for token dimension
):
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Initialize random state
    seed = batch_idx + 1  # Simple seed based on batch index
    rand = tl.rand(seed)
    
    # Mask for valid batch indices
    batch_mask = batch_idx < batch_size
    
    # Initialize cumulative probabilities
    cumsum = tl.zeros([BLOCK], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK], dtype=tl.int32)
    found = tl.zeros([BLOCK], dtype=tl.int32)
    
    # Process tokens in blocks
    for token_start in range(0, num_tokens, BLOCK_N):
        # Load token scores for current block
        token_idx = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_idx < num_tokens
        
        # Compute offsets for loading scores
        offsets = batch_idx[:, None] * scores_batch_stride + token_idx[None, :]
        block_scores = tl.load(scores_ptr + offsets, mask=batch_mask[:, None] & token_mask[None, :], other=0.0)
        
        # Compute cumulative sum for current block
        block_cumsum = tl.sum(block_scores, axis=1)
        cumsum += block_cumsum
        
        # Check if random value falls in current block
        block_mask = (rand[:] <= cumsum) & (found == 0)
        if tl.sum(block_mask) > 0:
            # Find exact position within block
            curr_sum = tl.zeros([BLOCK], dtype=tl.float32)
            for j in range(BLOCK_N):
                curr_sum += block_scores[:, j]
                update_mask = (rand <= curr_sum) & block_mask & (found == 0)
                selected_idx = tl.where(update_mask, token_start + j, selected_idx)
                found = tl.where(update_mask, 1, found)
    
    # Store results
    tl.store(output_ptr + batch_idx, selected_idx, mask=batch_mask)

def multinomial_sampling(scores: torch.Tensor) -> torch.Tensor:
    """
    Perform multinomial sampling on GPU using Triton.
    
    Args:
        scores: Tensor of shape [batch_size, num_tokens] containing unnormalized probabilities
        
    Returns:
        Tensor of shape [batch_size] containing sampled token indices
    """
    batch_size, num_tokens = scores.shape
    device = scores.device
    
    # Constants for block sizes
    BLOCK = 8
    BLOCK_N = 128
    
    # Compute grid size
    grid = (triton.cdiv(batch_size, BLOCK),)
    
    # Prepare output tensor
    output = torch.empty(batch_size, dtype=torch.int32, device=device)
    
    # Launch kernel
    _multinomial_sampling_kernel[grid](
        scores.contiguous().data_ptr(),
        output.data_ptr(),
        batch_size,
        num_tokens,
        scores.stride(0),
        BLOCK=BLOCK,
        BLOCK_N=BLOCK_N,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size = 32
    num_tokens = 1024
    scores = torch.rand(batch_size, num_tokens, device='cuda')
    scores = scores.softmax(dim=-1)  # Normalize to probabilities
    
    # Sample indices
    sampled_indices = multinomial_sampling(scores)
    print(f"Sampled indices shape: {sampled_indices.shape}")
