import torch
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,  # [total_cache_size, hidden_dim]
    cumsum_lengths_ptr,            # [num_seqs]
    cos_output_ptr, sin_output_ptr,# [total_length, hidden_dim]
    cache_stride,                  # scalar
    hidden_stride,                 # scalar
    total_length,                  # scalar
    HIDDEN_DIM: tl.constexpr,     # hidden dimension size
    N_ELEMENTS: tl.constexpr,     # total elements to process
    BLOCK_SIZE: tl.constexpr,     # processing block size
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    
    # Load sequence indices
    seq_idx = tl.load(cumsum_lengths_ptr + offsets, mask=mask)
    
    # Calculate original sequence position
    cache_pos = seq_idx % cache_stride
    hidden_idx = tl.arange(0, HIDDEN_DIM)
    
    # Load and store cache values
    for idx in range(0, BLOCK_SIZE):
        if idx < N_ELEMENTS:
            cos_cache_off = cache_pos[idx] * hidden_stride + hidden_idx
            sin_cache_off = cache_pos[idx] * hidden_stride + hidden_idx
            output_off = offsets[idx] * hidden_stride + hidden_idx
            
            cos_val = tl.load(cos_cache_ptr + cos_cache_off)
            sin_val = tl.load(sin_cache_ptr + sin_cache_off)
            
            tl.store(cos_output_ptr + output_off, cos_val)
            tl.store(sin_output_ptr + output_off, sin_val)

@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,  # [cache_size, hidden_dim]
    lengths_ptr,                   # [num_seqs]
    cos_output_ptr, sin_output_ptr,# [num_seqs, hidden_dim]
    cache_stride,                  # scalar
    hidden_stride,                 # scalar
    HIDDEN_DIM: tl.constexpr,     # hidden dimension size
    NUM_SEQS: tl.constexpr,       # number of sequences
    BLOCK_SIZE: tl.constexpr,     # processing block size
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_SEQS
    
    # Load sequence lengths
    seq_lengths = tl.load(lengths_ptr + offsets, mask=mask)
    hidden_idx = tl.arange(0, HIDDEN_DIM)
    
    # Process each sequence in the block
    for idx in range(0, BLOCK_SIZE):
        if idx < NUM_SEQS:
            cache_pos = (seq_lengths[idx] - 1) % cache_stride
            cos_cache_off = cache_pos * hidden_stride + hidden_idx
            sin_cache_off = cache_pos * hidden_stride + hidden_idx
            output_off = offsets[idx] * hidden_stride + hidden_idx
            
            cos_val = tl.load(cos_cache_ptr + cos_cache_off)
            sin_val = tl.load(sin_cache_ptr + sin_cache_off)
            
            tl.store(cos_output_ptr + output_off, cos_val)
            tl.store(sin_output_ptr + output_off, sin_val)

def get_sine_cache(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    lengths: torch.Tensor,
    is_prompts: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve cosine and sine cache values for sequence processing.
    
    Args:
        cos_cache: Cosine cache tensor [cache_size, hidden_dim]
        sin_cache: Sine cache tensor [cache_size, hidden_dim]
        lengths: Sequence lengths tensor [num_seqs]
        is_prompts: Whether processing prompts (True) or decoding (False)
    
    Returns:
        tuple of (cos_output, sin_output) tensors
    """
    device = cos_cache.device
    hidden_dim = cos_cache.shape[1]
    cache_size = cos_cache.shape[0]
    num_seqs = lengths.shape[0]
    
    # Calculate strides
    cache_stride = cache_size
    hidden_stride = hidden_dim
    
    if is_prompts:
        # Handle prompt processing
        cumsum_lengths = torch.cumsum(lengths, dim=0)
        total_length = cumsum_lengths[-1].item()
        
        cos_output = torch.empty((total_length, hidden_dim), device=device, dtype=cos_cache.dtype)
        sin_output = torch.empty((total_length, hidden_dim), device=device, dtype=sin_cache.dtype)
        
        grid = lambda meta: (triton.cdiv(total_length, meta['BLOCK_SIZE']),)
        
        prefill_cache_kernel[grid](
            cos_cache, sin_cache,
            cumsum_lengths,
            cos_output, sin_output,
            cache_stride,
            hidden_stride,
            total_length,
            HIDDEN_DIM=hidden_dim,
            N_ELEMENTS=total_length,
            BLOCK_SIZE=128,
        )
    else:
        # Handle decoding
        cos_output = torch.empty((num_seqs, hidden_dim), device=device, dtype=cos_cache.dtype)
        sin_output = torch.empty((num_seqs, hidden_dim), device=device, dtype=sin_cache.dtype)
        
        grid = lambda meta: (triton.cdiv(num_seqs, meta['BLOCK_SIZE']),)
        
        decoding_cache_kernel[grid](
            cos_cache, sin_cache,
            lengths,
            cos_output, sin_output,
            cache_stride,
            hidden_stride,
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=128,
        )
    
    return cos_output, sin_output
