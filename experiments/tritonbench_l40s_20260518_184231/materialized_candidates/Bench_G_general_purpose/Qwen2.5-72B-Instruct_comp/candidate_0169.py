import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    k_ptr,  # Pointer to the input key or value tensor
    cache_ptr,  # Pointer to the blocked cache
    batch_size,  # Batch size
    seq_len,  # Sequence length
    head_dim,  # Head dimension
    num_heads,  # Number of heads
    cache_block_size,  # Cache block size
    use_new_format,  # Flag to use new format
    BLOCK_SIZE: tl.constexpr,  # Block size for the kernel
):
    # Compute the batch and head indices
    pid = tl.program_id(axis=0)
    bid = pid // num_heads
    hid = pid % num_heads

    # Compute the current token and sequence indices
    current_token = tl.arange(0, BLOCK_SIZE)
    current_seq = current_token + bid * seq_len

    # Compute the past sequence lengths
    past_seq_len = current_seq - 1

    # Compute the cache block index
    cache_block_idx = current_seq // cache_block_size

    # Compute the cache offset within the block
    cache_offset = current_seq % cache_block_size

    # Compute the cache position
    if use_new_format:
        cache_pos = (bid * num_heads + hid) * cache_block_size * head_dim + cache_block_idx * head_dim + cache_offset * head_dim
    else:
        cache_pos = (bid * num_heads + hid) * seq_len * head_dim + past_seq_len * head_dim

    # Load the data from the input tensor
    k_pos = (bid * num_heads + hid) * seq_len * head_dim + current_token * head_dim
    k_data = tl.load(k_ptr + k_pos, mask=current_token < seq_len, other=0.0)

    # Store the data into the cache
    tl.store(cache_ptr + cache_pos, k_data, mask=current_token < seq_len)

import torch
import triton

def copy_k_to_blocked_cache(k, cache, batch_size, seq_len, head_dim, num_heads, cache_block_size, use_new_format):
    # Check input and cache dimensions
    assert k.shape == (batch_size, num_heads, seq_len, head_dim), "Input tensor 'k' has incorrect shape"
    assert cache.shape == (batch_size, num_heads, cache_block_size, head_dim), "Cache tensor has incorrect shape"

    # Compute grid and block dimensions
    grid = (batch_size * num_heads,)

    # Define the kernel launch configuration
    _copy_to_kcache_seqlen_n_kernel[grid](
        k,  # Pointer to the input key or value tensor
        cache,  # Pointer to the blocked cache
        batch_size,  # Batch size
        seq_len,  # Sequence length
        head_dim,  # Head dimension
        num_heads,  # Number of heads
        cache_block_size,  # Cache block size
        use_new_format,  # Flag to use new format
        BLOCK_SIZE=seq_len,  # Block size for the kernel
    )
