import torch
import triton
import triton.language as tl

@triton.jit
def _multinomial_sampling_kernel(
    Scores, Seeds, Offsets, Indices, Outputs,
    stride_sb, stride_st, stride_ib, stride_it,
    num_batchs, num_tokens,
    BLOCK: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get the batch block ID
    batch_block_id = tl.program_id(0)
    
    # Calculate offsets for current block
    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)
    
    # Create mask for valid batch indices
    off_mask = off < num_batchs
    
    # Load seeds and offsets for random number generation
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    
    # Generate random numbers for sampling
    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)
    
    # Process tokens in blocks
    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        
        # Load scores for current block
        scores = tl.load(
            Scores + off[:, None] * stride_sb + s_off[None, :] * stride_st,
            mask=s_mask,
            other=0.0
        ).to(tl.float32)
        
        # Calculate cumulative scores
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)
        
        # Find valid sampling positions
        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0
        
        # Get sampled indices
        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        indices = tl.load(
            Indices + off * stride_ib + valid_pos * stride_it,
            mask=found_mask & off_mask,
            other=-1
        )
        output = tl.where(found_mask, indices, output)
    
    # Store final results
    tl.store(Outputs + off, output, mask=off_mask)

def multinomial_sampling(
    scores: torch.Tensor,
    seeds: torch.LongTensor,
    offsets: torch.LongTensor,
    indices: torch.Tensor = None
):
    """
    Perform multinomial sampling on GPU using Triton.
    
    Args:
        scores: Input tensor of shape [batch_size, num_tokens]
        seeds: Random seeds for sampling
        offsets: Offsets for random number generation
        indices: Optional tensor for custom token indices
    
    Returns:
        torch.Tensor: Sampled indices for each batch
    """
    assert scores.dim() == 2
    batch_size, num_tokens = scores.size()
    device = scores.device
    
    # Handle special case of single token
    if num_tokens == 1:
        return torch.zeros_like(scores, dtype=torch.long)
    
    # Create default indices if not provided
    if indices is None:
        indices = torch.arange(num_tokens, device=device)
        indices = indices.expand_as(scores)
    
    assert indices.dim() == 2
    assert indices.size() == scores.size()
    
    # Initialize output tensor
    outputs = indices[:, 0].clone()
    
    # Define block sizes
    BLOCK = 8
    BLOCK_N = 128
    
    # Launch kernel
    grid = (triton.cdiv(batch_size, BLOCK),)
    _multinomial_sampling_kernel[grid](
        scores,
        seeds,
        offsets,
        indices,
        outputs,
        stride_sb=scores.stride(0),
        stride_st=scores.stride(1),
        stride_ib=indices.stride(0),
        stride_it=indices.stride(1),
        num_batchs=batch_size,
        num_tokens=num_tokens,
        BLOCK=BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=8
    )
    
    return outputs
