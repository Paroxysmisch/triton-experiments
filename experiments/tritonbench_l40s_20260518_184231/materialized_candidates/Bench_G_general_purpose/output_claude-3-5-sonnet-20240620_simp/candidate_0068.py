import triton
import triton.language as tl
import torch

@triton.jit
def _multinomial_sampling_kernel(
    scores_ptr,        # pointer to scores tensor [batch_size, num_tokens]
    seeds_ptr,         # pointer to random seeds tensor [batch_size]
    offsets_ptr,       # pointer to offsets tensor [batch_size]
    output_ptr,        # pointer to output tensor [batch_size]
    batch_size,        # number of sequences in batch
    num_tokens,        # vocabulary size
    BLOCK_SIZE: tl.constexpr,  # size of parallel processing block
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Handle one sequence per thread block
    if pid >= batch_size:
        return
        
    # Load random seed and offset for this sequence
    seed = tl.load(seeds_ptr + pid)
    offset = tl.load(offsets_ptr + pid)
    
    # Initialize random state
    rand_state = tl.rand.PhiloxRandomState(seed=seed, offset=offset)
    
    # Generate random value for sampling
    rand_val = tl.rand.uniform(rand_state)
    
    # Calculate cumulative probabilities
    cumsum = 0.0
    selected_token = num_tokens - 1  # default to last token
    
    # Process tokens in blocks for better memory access
    for block_start in range(0, num_tokens, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, num_tokens)
        
        # Load scores for current block
        scores = tl.load(scores_ptr + pid * num_tokens + 
                        tl.arange(0, block_end - block_start) + block_start)
        
        # Convert to probabilities using softmax
        scores = tl.exp(scores - tl.max(scores))
        probs = scores / tl.sum(scores)
        
        # Update cumulative sum and check for sampling position
        for i in range(block_end - block_start):
            cumsum += probs[i]
            # If random value falls in this probability range
            selected_token = tl.where(
                (rand_val <= cumsum) & (selected_token == num_tokens - 1),
                block_start + i,
                selected_token
            )
    
    # Store result
    tl.store(output_ptr + pid, selected_token)

def multinomial_sampling(scores, seeds, offsets):
    """
    Wrapper function for multinomial sampling using Triton kernel
    
    Args:
        scores: torch.Tensor, shape [batch_size, num_tokens] - input scores
        seeds: torch.Tensor, shape [batch_size] - random seeds
        offsets: torch.Tensor, shape [batch_size] - random state offsets
    
    Returns:
        torch.Tensor, shape [batch_size] - sampled token indices
    """
    batch_size, num_tokens = scores.shape
    device = scores.device
    
    # Allocate output tensor
    output = torch.empty(batch_size, dtype=torch.int32, device=device)
    
    # Configure kernel parameters
    BLOCK_SIZE = 128
    grid = (batch_size,)
    
    # Launch kernel
    _multinomial_sampling_kernel[grid](
        scores.contiguous().data_ptr(),
        seeds.contiguous().data_ptr(),
        offsets.contiguous().data_ptr(),
        output.data_ptr(),
        batch_size,
        num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
