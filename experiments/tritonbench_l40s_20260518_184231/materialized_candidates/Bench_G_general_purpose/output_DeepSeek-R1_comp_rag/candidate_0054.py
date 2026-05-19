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
    # Determine batch block index and offsets
    batch_block_id = tl.program_id(0)
    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)
    
    # Mask for valid batches in this block
    off_mask = off < num_batchs
    
    # Load seeds and offsets for random number generation
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)
    
    # Generate random samples (one per batch)
    samp = tl.rand(seed, offset)[:, None]
    
    # Initialize accumulation and output
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    # Process tokens in blocks of BLOCK_N
    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        
        # Load scores block
        scores = tl.load(
            Scores + off[:, None] * stride_sb + s_off[None, :] * stride_st,
            mask=s_mask, other=0.0
        ).to(tl.float32)
        
        # Compute cumulative scores
        c_scores = tl.cumsum(scores, axis=1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, axis=1)
        
        # Find where sample falls in cumulative distribution
        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask.to(tl.int1), axis=1) > 0
        
        # Get corresponding indices
        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), axis=1)
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
) -> torch.Tensor:
    """
    Performs multinomial sampling using Triton kernel.
    
    Args:
        scores: 2D tensor of shape [batch_size, num_tokens]
        seeds: 1D tensor of random seeds per batch
        offsets: 1D tensor of offset counters for random generation
        indices: Optional 2D tensor of token indices (for sorted distributions)
    
    Returns:
        Sampled indices tensor of shape [batch_size]
    """
    assert scores.dim() == 2
    batch_size, num_tokens = scores.shape
    device = scores.device

    # Handle edge case with single token
    if num_tokens == 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    # Create default indices if not provided
    if indices is None:
        indices = torch.arange(num_tokens, device=device).expand_as(scores)
    assert indices.size() == scores.size()

    # Initialize output with first token indices
    outputs = indices[:, 0].clone()

    # Kernel configuration
    BLOCK = 8
    BLOCK_N = 128
    grid = (triton.cdiv(batch_size, BLOCK),)

    # Launch kernel
    _multinomial_sampling_kernel[grid](
        Scores=scores,
        Seeds=seeds,
        Offsets=offsets,
        Indices=indices,
        Outputs=outputs,
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
