import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K_ptr, V_ptr, KCache_ptr, VCache_ptr, block_tables_ptr, context_lengths_ptr,
    num_blocks, num_kv_heads, block_size, head_dim, x,
    K_strides, V_strides, KCache_strides, VCache_strides,
    new_cache_layout: tl.constexpr
):
    # Get the current program ID
    pid = tl.program_id(0)
    num_heads = num_kv_heads

    # Compute the sequence and head indices
    seq_idx = pid // num_heads
    head_idx = pid % num_heads

    # Get the block table for the current sequence
    block_table_offset = seq_idx * num_blocks
    block_table = tl.load(block_tables_ptr + block_table_offset, num_blocks)

    # Get the context length for the current sequence
    context_length = tl.load(context_lengths_ptr + seq_idx)

    # Compute the offset in the cache
    if new_cache_layout:
        # Five-dimensional layout: [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        KCache_offset = (block_table * num_heads + head_idx) * (head_dim // x) * block_size * x
        VCache_offset = (block_table * num_heads + head_idx) * (head_dim // x) * block_size * x
    else:
        # Four-dimensional layout: [num_blocks, num_kv_heads, block_size, head_dim]
        KCache_offset = (block_table * num_heads + head_idx) * block_size * head_dim
        VCache_offset = (block_table * num_heads + head_idx) * block_size * head_dim

    # Compute the offset in the input tensors
    K_offset = (seq_idx * K_strides[0] + head_idx * K_strides[1]) * head_dim
    V_offset = (seq_idx * V_strides[0] + head_idx * V_strides[1]) * head_dim

    # Load the data from the input tensors
    K_data = tl.load(K_ptr + K_offset, head_dim)
    V_data = tl.load(V_ptr + V_offset, head_dim)

    # Store the data in the cache
    tl.store(KCache_ptr + KCache_offset, K_data)
    tl.store(VCache_ptr + VCache_offset, V_data)

import torch
import triton
import triton.language as tl

def copy_kv_to_blocked_cache(K, V, KCache, VCache, block_tables, context_lengths, num_blocks, num_kv_heads, block_size, head_dim, x, new_cache_layout=False):
    # Assert input shapes
    assert K.shape == (K.size(0), num_kv_heads, head_dim), "K tensor shape mismatch"
    assert V.shape == (V.size(0), num_kv_heads, head_dim), "V tensor shape mismatch"
    assert KCache.shape == (num_blocks, num_kv_heads, block_size, head_dim) or KCache.shape == (num_blocks, num_kv_heads, head_dim // x, block_size, x), "KCache tensor shape mismatch"
    assert VCache.shape == (num_blocks, num_kv_heads, block_size, head_dim) or VCache.shape == (num_blocks, num_kv_heads, head_dim // x, block_size, x), "VCache tensor shape mismatch"
    assert block_tables.shape == (K.size(0), num_blocks), "block_tables tensor shape mismatch"
    assert context_lengths.shape == (K.size(0),), "context_lengths tensor shape mismatch"

    # Calculate strides
    K_strides = (K.stride(0), K.stride(1))
    V_strides = (V.stride(0), V.stride(1))

    if new_cache_layout:
        # Five-dimensional layout: [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        KCache_strides = (KCache.stride(0), KCache.stride(1), KCache.stride(2), KCache.stride(3))
        VCache_strides = (VCache.stride(0), VCache.stride(1), VCache.stride(2), VCache.stride(3))
    else:
        # Four-dimensional layout: [num_blocks, num_kv_heads, block_size, head_dim]
        KCache_strides = (KCache.stride(0), KCache.stride(1), KCache.stride(2))
        VCache_strides = (VCache.stride(0), VCache.stride(1), VCache.stride(2))

    # Calculate grid size
    grid_size = (K.size(0) * num_kv_heads,)

    # Launch the kernel
    _copy_to_kvcache_seqlen1_kernel[grid_size](
        K, V, KCache, VCache, block_tables, context_lengths,
        num_blocks, num_kv_heads, block_size, head_dim, x,
        K_strides, V_strides, KCache_strides, VCache_strides,
        new_cache_layout
    )
