import triton
import triton.language as tl

def copy_kv_to_blocked_cache(
    K, V, KCache, VCache, block_tables, context_lengths,
    stride_KKV, stride_KKVHead, stride_KKVBlock, stride_KKVHeadDim,
    stride_KVCache, stride_KVCacheBlock, stride_KVCacheHeadDim, stride_KVCacheHeadDimX,
    block_size, head_dim, head_dim_x, num_heads, batch_size, cache_layout
):
    # Check input shapes
    assert K.shape == (batch_size, num_heads, block_size, head_dim)
    assert V.shape == (batch_size, num_heads, block_size, head_dim)
    assert KCache.shape == (batch_size, num_heads, block_size, head_dim)
    assert VCache.shape == (batch_size, num_heads, block_size, head_dim)
    assert block_tables.shape == (batch_size, num_heads)
    assert context_lengths.shape == (batch_size,)

    # Calculate strides and offsets based on cache layout
    if cache_layout == 0:  # 4D layout
        stride_KVCacheHeadDim = head_dim
    elif cache_layout == 1:  # 5D layout
        stride_KVCacheHeadDimX = head_dim // head_dim_x
        stride_KVCacheHeadDim = head_dim
    else:
        raise ValueError("Unsupported cache layout")

    # Launch the kernel
    grid = (batch_size * num_heads,)
    block = (block_size,)
    _copy_to_kvcache_seqlen1_kernel[grid, block](
        K, V, KCache, VCache, block_tables, context_lengths,
        stride_KKV, stride_KKVHead, stride_KKVBlock, stride_KKVHeadDim,
        stride_KVCache, stride_KVCacheBlock, stride_KVCacheHeadDim, stride_KVCacheHeadDimX,
        block_size, head_dim, head_dim_x, num_heads, batch_size, cache_layout
    )
