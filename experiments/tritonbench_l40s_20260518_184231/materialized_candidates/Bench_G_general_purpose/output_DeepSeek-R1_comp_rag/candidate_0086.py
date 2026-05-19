import torch
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr,
    sin_cache_ptr,
    cumsum_lengths_ptr,
    cos_output_ptr,
    sin_output_ptr,
    cache_stride,
    hidden_stride,
    total_length,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_length
    
    # Binary search to find sequence index for each offset
    seq_idx = tl.zeros(BLOCK_SIZE, dtype=tl.int32)
    for i in range(0, tl.math.log2(NUM_SEQS).to(tl.int32) + 1):
        bit = NUM_SEQS >> (i + 1)
        compare = tl.load(cumsum_lengths_ptr + seq_idx + bit) <= offs
        seq_idx += tl.where(compare, bit, 0)
    
    # Get position within sequence
    cumsum = tl.load(cumsum_lengths_ptr + seq_idx, mask=mask, other=0)
    pos = offs - cumsum
    
    # Load cache entries
    cache_offsets = pos[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride
    cos_vals = tl.load(cos_cache_ptr + cache_offsets, mask=mask[:, None], other=0)
    sin_vals = tl.load(sin_cache_ptr + cache_offsets, mask=mask[:, None], other=0)
    
    # Store results
    output_offsets = offs[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride
    tl.store(cos_output_ptr + output_offsets, cos_vals, mask=mask[:, None])
    tl.store(sin_output_ptr + output_offsets, sin_vals, mask=mask[:, None])

@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr,
    sin_cache_ptr,
    lengths_ptr,
    cos_output_ptr,
    sin_output_ptr,
    cache_stride,
    hidden_stride,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUM_SEQS
    
    # Load lengths and compute positions
    seq_pos = tl.load(lengths_ptr + offs, mask=mask, other=0)
    
    # Load cache entries
    cache_offsets = seq_pos[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride
    cos_vals = tl.load(cos_cache_ptr + cache_offsets, mask=mask[:, None], other=0)
    sin_vals = tl.load(sin_cache_ptr + cache_offsets, mask=mask[:, None], other=0)
    
    # Store results
    output_offsets = offs[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride
    tl.store(cos_output_ptr + output_offsets, cos_vals, mask=mask[:, None])
    tl.store(sin_output_ptr + output_offsets, sin_vals, mask=mask[:, None])

def get_xine_cache(lengths: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor, is_prompts: bool = False):
    assert cos_cache.shape == sin_cache.shape
    _, hidden_dim = cos_cache.shape
    num_seqs = lengths.numel()

    # Determine optimal launch parameters
    if hidden_dim >= 256:
        num_warps = 16
    elif hidden_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    cache_stride = cos_cache.stride(0)
    hidden_stride = cos_cache.stride(1)

    if is_prompts:
        BLOCK_SIZE = 128
        total_length = lengths.sum().item()
        cumsum_lens = torch.cumsum(lengths, dim=0)
        cos_output = torch.empty((total_length, hidden_dim), dtype=cos_cache.dtype, device=cos_cache.device)
        sin_output = torch.empty((total_length, hidden_dim), dtype=sin_cache.dtype, device=sin_cache.device)
        
        grid = (triton.cdiv(total_length, BLOCK_SIZE),)
        prefill_cache_kernel[grid](
            cos_cache, sin_cache, cumsum_lens,
            cos_output, sin_output,
            cache_stride, hidden_stride,
            total_length,
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        BLOCK_SIZE = 128
        nlengths = lengths.to(torch.int32) - 1
        cos_output = torch.empty((num_seqs, hidden_dim), dtype=cos_cache.dtype, device=cos_cache.device)
        sin_output = torch.empty((num_seqs, hidden_dim), dtype=sin_cache.dtype, device=sin_cache.device)
        
        grid = (triton.cdiv(num_seqs, BLOCK_SIZE),)
        decoding_cache_kernel[grid](
            cos_cache, sin_cache, nlengths,
            cos_output, sin_output,
            cache_stride, hidden_stride,
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return cos_output, sin_output
