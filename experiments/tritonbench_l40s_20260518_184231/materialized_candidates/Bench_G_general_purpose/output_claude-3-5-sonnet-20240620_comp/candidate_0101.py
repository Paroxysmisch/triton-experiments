import triton
import triton.language as tl
import torch

@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,  # input cache pointers
    cos_output_ptr, sin_output_ptr,  # output cache pointers
    cumsum_lengths_ptr,  # cumulative sum of sequence lengths
    total_length,  # total sequence length
    HIDDEN_DIM: tl.constexpr,  # hidden dimension size
    N_ELEMENTS: tl.constexpr,  # number of elements in cache
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < total_length
    
    # Load cumsum lengths and calculate original sequence indices
    seq_indices = tl.load(cumsum_lengths_ptr + offsets, mask=mask)
    
    # Load from cache and store to output
    for h in range(0, HIDDEN_DIM, BLOCK_SIZE):
        h_offsets = h + tl.arange(0, BLOCK_SIZE)
        h_mask = h_offsets < HIDDEN_DIM
        
        # Calculate source indices
        src_indices = seq_indices * HIDDEN_DIM + h_offsets
        dst_indices = offsets * HIDDEN_DIM + h_offsets
        
        # Load and store cosine cache
        cos_values = tl.load(cos_cache_ptr + src_indices, mask=mask & h_mask)
        tl.store(cos_output_ptr + dst_indices, cos_values, mask=mask & h_mask)
        
        # Load and store sine cache
        sin_values = tl.load(sin_cache_ptr + src_indices, mask=mask & h_mask)
        tl.store(sin_output_ptr + dst_indices, sin_values, mask=mask & h_mask)

@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,  # input cache pointers
    cos_output_ptr, sin_output_ptr,  # output cache pointers
    lengths_ptr,  # sequence lengths
    HIDDEN_DIM: tl.constexpr,  # hidden dimension size
    NUM_SEQS: tl.constexpr,  # number of sequences
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    # Get program ID for sequence
    pid = tl.program_id(0)
    
    # Exit if beyond number of sequences
    if pid >= NUM_SEQS:
        return
        
    # Load sequence length
    seq_len = tl.load(lengths_ptr + pid) - 1  # subtract 1 to get previous position
    
    # Process hidden dimensions in blocks
    for h in range(0, HIDDEN_DIM, BLOCK_SIZE):
        h_offsets = h + tl.arange(0, BLOCK_SIZE)
        h_mask = h_offsets < HIDDEN_DIM
        
        # Calculate source and destination indices
        src_indices = seq_len * HIDDEN_DIM + h_offsets
        dst_indices = pid * HIDDEN_DIM + h_offsets
        
        # Load and store cosine cache
        cos_values = tl.load(cos_cache_ptr + src_indices, mask=h_mask)
        tl.store(cos_output_ptr + dst_indices, cos_values, mask=h_mask)
        
        # Load and store sine cache
        sin_values = tl.load(sin_cache_ptr + src_indices, mask=h_mask)
        tl.store(sin_output_ptr + dst_indices, sin_values, mask=h_mask)

def get_xine_cache(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    lengths: torch.Tensor,
    is_prompt: bool = False,
    block_size: int = 128
):
    """
    Wrapper function to handle both prefill and decoding cache operations.
    
    Args:
        cos_cache: Cosine cache tensor
        sin_cache: Sine cache tensor
        lengths: Sequence lengths tensor
        is_prompt: Flag to determine if processing prompts
        block_size: Block size for parallelization
    """
    hidden_dim = cos_cache.shape[-1]
    
    if is_prompt:
        # Handle prompt processing
        total_length = lengths.sum().item()
        cumsum_lengths = torch.cumsum(lengths, dim=0) - 1  # -1 for 0-based indexing
        
        # Prepare output tensors
        cos_output = torch.empty((total_length, hidden_dim), 
                               device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty((total_length, hidden_dim), 
                               device=sin_cache.device, dtype=sin_cache.dtype)
        
        # Launch prefill kernel
        grid = lambda meta: (triton.cdiv(total_length, block_size),)
        prefill_cache_kernel[grid](
            cos_cache.data_ptr(), sin_cache.data_ptr(),
            cos_output.data_ptr(), sin_output.data_ptr(),
            cumsum_lengths.data_ptr(),
            total_length,
            hidden_dim,
            cos_cache.shape[0],
            block_size
        )
    else:
        # Handle decoding
        num_seqs = lengths.shape[0]
        
        # Prepare output tensors
        cos_output = torch.empty((num_seqs, hidden_dim), 
                               device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty((num_seqs, hidden_dim), 
                               device=sin_cache.device, dtype=sin_cache.dtype)
        
        # Launch decoding kernel
        grid = lambda meta: (num_seqs,)
        decoding_cache_kernel[grid](
            cos_cache.data_ptr(), sin_cache.data_ptr(),
            cos_output.data_ptr(), sin_output.data_ptr(),
            lengths.data_ptr(),
            hidden_dim,
            num_seqs,
            block_size
        )
    
    return cos_output, sin_output
