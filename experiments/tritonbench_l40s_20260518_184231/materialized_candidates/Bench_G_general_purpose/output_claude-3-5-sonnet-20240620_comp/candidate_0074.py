import triton
import triton.language as tl
import torch

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    # Pointers to input tensors
    K_ptr, V_ptr,
    # Pointers to cache tensors 
    KCache_ptr, VCache_ptr,
    # Block mapping and context info
    block_tables_ptr, context_lengths_ptr,
    # Dimensions
    head_dim, block_size, num_kv_heads,
    # Strides for memory access
    K_batch_stride, K_head_stride,
    KCache_block_stride, KCache_head_stride, KCache_dim_stride,
    # Whether using new cache layout
    use_5d_layout: tl.constexpr,
    x_dim: tl.constexpr,
):
    # Get sequence and head indices
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Load context length and block index
    ctx_len = tl.load(context_lengths_ptr + seq_idx)
    block_idx = tl.load(block_tables_ptr + seq_idx)
    
    # Calculate source offsets
    k_offset = seq_idx * K_batch_stride + head_idx * K_head_stride
    v_offset = k_offset  # Same layout for K and V
    
    # Calculate cache offsets
    if use_5d_layout:
        # 5D layout: [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        cache_offset = (block_idx * KCache_block_stride + 
                       head_idx * KCache_head_stride)
        
        # Load and store in tiles of size x
        for dim_idx in range(0, head_dim, x_dim):
            dim_offset = dim_idx // x_dim * KCache_dim_stride
            
            # Load x elements from K and V
            k_vals = tl.load(K_ptr + k_offset + dim_idx + tl.arange(0, x_dim))
            v_vals = tl.load(V_ptr + v_offset + dim_idx + tl.arange(0, x_dim))
            
            # Store to cache
            k_cache_ptr = KCache_ptr + cache_offset + dim_offset
            v_cache_ptr = VCache_ptr + cache_offset + dim_offset
            tl.store(k_cache_ptr + tl.arange(0, x_dim), k_vals)
            tl.store(v_cache_ptr + tl.arange(0, x_dim), v_vals)
    else:
        # 4D layout: [num_blocks, num_kv_heads, block_size, head_dim]
        cache_offset = (block_idx * KCache_block_stride + 
                       head_idx * KCache_head_stride)
        
        # Load and store entire head dimension at once
        k_vals = tl.load(K_ptr + k_offset + tl.arange(0, head_dim))
        v_vals = tl.load(V_ptr + v_offset + tl.arange(0, head_dim))
        
        # Store to cache
        tl.store(KCache_ptr + cache_offset + tl.arange(0, head_dim), k_vals)
        tl.store(VCache_ptr + cache_offset + tl.arange(0, head_dim), v_vals)

def copy_kv_to_blocked_cache(
    K: torch.Tensor,
    V: torch.Tensor,
    KCache: torch.Tensor,
    VCache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_size: int,
    x_dim: int = 8,  # For 5D layout, typically 8 or 16
):
    batch_size, num_kv_heads, head_dim = K.shape
    
    # Verify shapes
    assert K.shape == V.shape
    assert KCache.shape == VCache.shape
    assert block_tables.shape[0] == batch_size
    assert context_lengths.shape[0] == batch_size
    
    # Determine cache layout
    use_5d_layout = len(KCache.shape) == 5
    if use_5d_layout:
        assert head_dim % x_dim == 0, f"head_dim {head_dim} must be divisible by x_dim {x_dim}"
    
    # Calculate strides
    K_batch_stride = K.stride(0)
    K_head_stride = K.stride(1)
    
    if use_5d_layout:
        KCache_block_stride = KCache.stride(0)
        KCache_head_stride = KCache.stride(1)
        KCache_dim_stride = KCache.stride(2) * x_dim
    else:
        KCache_block_stride = KCache.stride(0)
        KCache_head_stride = KCache.stride(1)
        KCache_dim_stride = 0  # Not used in 4D layout
    
    # Launch kernel
    grid = (batch_size, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        K, V,
        KCache, VCache,
        block_tables, context_lengths,
        head_dim, block_size, num_kv_heads,
        K_batch_stride, K_head_stride,
        KCache_block_stride, KCache_head_stride, KCache_dim_stride,
        use_5d_layout, x_dim,
    )
