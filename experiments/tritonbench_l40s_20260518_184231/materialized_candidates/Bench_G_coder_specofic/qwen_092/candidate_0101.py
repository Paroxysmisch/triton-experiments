triton
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache: tl.tensor, sin_cache: tl.tensor,
    cos_output: tl.tensor, sin_output: tl.tensor,
    cumsum_lengths: tl.tensor,
    HIDDEN_DIM: tl.constexpr, N_ELEMENTS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Prefills the cache for prompts.
    """
    # Get the current block and grid indices
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Compute the global index within the cache
    global_index = row * HIDDEN_DIM + col

    # Calculate the original sequence index for the target index
    target_index = row * BLOCK_SIZE + col
    seq_index = tl.searchsorted(cumsum_lengths, target_index, side='right') - 1
    offset = target_index - cumsum_lengths[seq_index]

    # Copy the relevant parts from the cache to the output
    cos_output[global_index] = cos_cache[seq_index * HIDDEN_DIM + col]
    sin_output[global_index] = sin_cache[seq_index * HIDDEN_DIM + col]

@triton.jit
def decoding_cache_kernel(
    cos_cache: tl.tensor, sin_cache: tl.tensor,
    cos_output: tl.tensor, sin_output: tl.tensor,
    lengths: tl.tensor,
    HIDDEN_DIM: tl.constexpr, NUM_SEQS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Decodes the cache based on sequence lengths.
    """
    # Get the current block and grid indices
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Compute the global index within the cache
    global_index = row * HIDDEN_DIM + col

    # Calculate the previous cache entry index
    seq_index = row // BLOCK_SIZE
    prev_seq_index = seq_index - 1
    offset = row % BLOCK_SIZE

    # Copy the relevant parts from the cache to the output
    cos_output[global_index] = cos_cache[prev_seq_index * HIDDEN_DIM + col]
    sin_output[global_index] = sin_cache[prev_seq_index * HIDDEN_DIM + col]

@triton.jit
def get_xine_cache(
    cos_cache: tl.tensor, sin_cache: tl.tensor,
    cos_output: tl.tensor, sin_output: tl.tensor,
    lengths: tl.tensor,
    cumsum_lengths: tl.tensor,
    is_prompts: tl.constexpr,
    HIDDEN_DIM: tl.constexpr, N_ELEMENTS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    NUM_SEQS: tl.constexpr
):
    """
    Determines which kernel to execute based on the is_prompts flag.
    """
    if is_prompts:
        # Prefill cache for prompts
        num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks, NUM_SEQS)
        prefill_cache_kernel[grid](cos_cache, sin_cache, cos_output, sin_output, cumsum_lengths, HIDDEN_DIM, N_ELEMENTS, BLOCK_SIZE)
    else:
        # Decode cache based on sequence lengths
        num_blocks = (N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks, NUM_SEQS)
        decoding_cache_kernel[grid](cos_cache, sin_cache, cos_output, sin_output, lengths, HIDDEN_DIM, NUM_SEQS, BLOCK_SIZE)

# Example usage
# cos_cache = tl.zeros((NUM_SEQS * HIDDEN_DIM,), dtype=tl.float32)
# sin_cache = tl.zeros((NUM_SEQS * HIDDEN_DIM,), dtype=tl.float32)
# cos_output = tl.zeros((N_ELEMENTS * HIDDEN_DIM,), dtype=tl.float32)
# sin_output = tl.zeros((N_ELEMENTS * HIDDEN_DIM,), dtype=tl.float32)
# lengths = tl.zeros((NUM_SEQS,), dtype=tl.int32)
# cumsum_lengths = tl.zeros((NUM_SEQS,), dtype=tl.int32)
# is_prompts = tl.constant(True, dtype=tl.bool)
# get_xine_cache(cos_cache, sin_cache, cos_output, sin_output, lengths, cumsum_lengths, is_prompts, HIDDEN_DIM, N_ELEMENTS, BLOCK_SIZE, NUM_SEQS)
