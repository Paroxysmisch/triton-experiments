import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    K_ptr,  # Pointer to the input tensor K
    KCache_ptr,  # Pointer to the pre-allocated cache
    block_table_ptr,  # Pointer to the block table
    seq_lengths_ptr,  # Pointer to the sequence lengths
    n_tokens,  # Number of tokens per sequence
    n_heads,  # Number of attention heads
    head_dim,  # Dimension of each head
    cache_block_size,  # Size of each cache block
    seq_len,  # Length of the sequence
    BLOCK_M: tl.constexpr,  # Block size for the sequence length
    BLOCK_D: tl.constexpr,  # Block size for the head dimension
):
    # Get the current block ID
    pid = tl.program_id(axis=0)
    
    # Compute the starting position in the sequence
    start_seq = pid * BLOCK_M
    end_seq = tl.minimum(start_seq + BLOCK_M, seq_len)
    
    # Iterate over the sequences
    for seq in range(start_seq, end_seq):
        # Load the sequence length
        seq_length = tl.load(seq_lengths_ptr + seq)
        
        # Compute the block ID in the block table
        block_id = seq // n_tokens
        
        # Compute the starting position in the cache
        cache_start = tl.load(block_table_ptr + block_id) * cache_block_size
        
        # Iterate over the tokens in the sequence
        for token in range(seq % n_tokens):
            # Compute the position in the input tensor K
            k_pos = (seq * n_tokens + token) * n_heads * head_dim
            
            # Compute the position in the cache
            cache_pos = (cache_start + seq % n_tokens + token) * n_heads * head_dim
            
            # Load the data from K
            k_data = tl.load(K_ptr + k_pos, mask=token < seq_length, other=0.0)
            
            # Store the data in the cache
            tl.store(KCache_ptr + cache_pos, k_data, mask=token < seq_length)

import torch

def copy_k_to_blocked_cache(K, KCache, block_table, seq_lengths, n_tokens, n_heads, head_dim, cache_block_size, seq_len, BLOCK_M=128, BLOCK_D=64):
    # Check input dimensions
    assert K.shape == (seq_len, n_tokens, n_heads, head_dim), "Input tensor K has incorrect shape"
    assert KCache.shape == (seq_len, cache_block_size, n_heads, head_dim), "Cache tensor KCache has incorrect shape"
    assert block_table.shape == (seq_len // n_tokens,), "Block table has incorrect shape"
    assert seq_lengths.shape == (seq_len,), "Sequence lengths have incorrect shape"
    
    # Determine the number of warps
    num_warps = 4 if head_dim <= 2048 else 8
    
    # Launch the kernel
    grid = (seq_len // BLOCK_M + (seq_len % BLOCK_M > 0),)
    _copy_to_kcache_seqlen_n_kernel[grid](
        K, KCache, block_table, seq_lengths, n_tokens, n_heads, head_dim, cache_block_size, seq_len, BLOCK_M, BLOCK_D,
        num_warps=num_warps
    )
