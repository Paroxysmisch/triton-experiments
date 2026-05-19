import triton
import triton.language as tl

# Constants
HIDDEN_DIM = 1024  # Example hidden dimension size
N_ELEMENTS = 128   # Number of elements to process per block
BLOCK_SIZE = 128   # Block size for parallel processing

# Kernel to pre-fill caches for prompts
@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr, sin_cache_ptr, cos_output_ptr, sin_output_ptr,
    cumsum_lengths_ptr, total_length, HIDDEN_DIM, N_ELEMENTS, BLOCK_SIZE,
    grid=(1,), num_warps=4
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_length

    # Load cumulative sum of lengths
    cumsum_lengths = tl.load(cumsum_lengths_ptr + offsets, mask=mask, other=0)

    # Calculate the original sequence index for each target index
    seq_indices = tl.where(offsets < cumsum_lengths, offsets, cumsum_lengths - 1)

    # Load the cache values
    cos_cache = tl.load(cos_cache_ptr + seq_indices * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), mask=mask, other=0)
    sin_cache = tl.load(sin_cache_ptr + seq_indices * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), mask=mask, other=0)

    # Store the cache values in the output buffers
    tl.store(cos_output_ptr + offsets * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), cos_cache, mask=mask)
    tl.store(sin_output_ptr + offsets * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), sin_cache, mask=mask)

# Kernel to decode cache data based on sequence lengths
@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr, sin_cache_ptr, cos_output_ptr, sin_output_ptr,
    lengths_ptr, NUM_SEQS, HIDDEN_DIM, BLOCK_SIZE,
    grid=(1,), num_warps=4
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_SEQS

    # Load sequence lengths
    lengths = tl.load(lengths_ptr + offsets, mask=mask, other=0)

    # Calculate the previous cache index for each sequence
    prev_indices = lengths - 1

    # Load the cache values
    cos_cache = tl.load(cos_cache_ptr + prev_indices * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), mask=mask, other=0)
    sin_cache = tl.load(sin_cache_ptr + prev_indices * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), mask=mask, other=0)

    # Store the cache values in the output buffers
    tl.store(cos_output_ptr + offsets * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), cos_cache, mask=mask)
    tl.store(sin_output_ptr + offsets * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), sin_cache, mask=mask)

# Utility function to determine which kernel to execute
def get_xine_cache(cos_cache, sin_cache, cos_output, sin_output, lengths, is_prompts):
    if is_prompts:
        total_length = lengths.sum()
        cumsum_lengths = lengths.cumsum()
        grid = (total_length // BLOCK_SIZE + 1,)
        prefill_cache_kernel[grid](cos_cache, sin_cache, cos_output, sin_output, cumsum_lengths, total_length, HIDDEN_DIM, N_ELEMENTS, BLOCK_SIZE)
    else:
        num_seqs = len(lengths)
        grid = (num_seqs // BLOCK_SIZE + 1,)
        decoding_cache_kernel[grid](cos_cache, sin_cache, cos_output, sin_output, lengths, num_seqs, HIDDEN_DIM, BLOCK_SIZE)
