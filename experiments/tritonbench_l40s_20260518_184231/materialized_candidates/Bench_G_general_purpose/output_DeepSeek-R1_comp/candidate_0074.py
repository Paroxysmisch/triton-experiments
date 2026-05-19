import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    # Input tensors
    K, V,
    # Cache tensors
    KCache, VCache,
    # Block tables and context lengths
    BlockTables, ContextLengths,
    # Strides for K and V
    stride_k_b, stride_k_h, stride_k_d,
    stride_v_b, stride_v_h, stride_v_d,
    # KCache strides: [b, h, ...]
    stride_kcache_b, stride_kcache_h,
    # For 4D layout: [b, h, bs, d] - stride_bs and stride_d
    # For 5D layout: [b, h, d0, bs, x] - stride_d0, stride_bs, stride_x
    stride_kcache_bs, stride_kcache_d,
    stride_kcache_d0, stride_kcache_x,
    # VCache strides: similar to KCache
    stride_vcache_b, stride_vcache_h,
    stride_vcache_bs, stride_vcache_d,
    stride_vcache_d0, stride_vcache_x,
    # Block tables strides
    stride_blocktables_s, stride_blocktables_b,
    # Parameters
    head_dim,
    block_size: tl.constexpr,
    new_cache_layout: tl.constexpr,
    x: tl.constexpr  # divisor for 5D layout
):
    # 2D grid over [batch_size, num_heads]
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Bounds checking
    batch_size = tl.num_programs(0)
    num_heads = tl.num_programs(1)
    if pid_seq >= batch_size or pid_head >= num_heads:
        return
    
    # Load context length for current sequence
    context_len_ptr = ContextLengths + pid_seq
    context_len = tl.load(context_len_ptr)
    
    # Calculate position in block and block index
    pos_in_block = context_len % block_size
    block_idx = context_len // block_size
    
    # Get physical block number from block table
    block_table_ptr = BlockTables + pid_seq * stride_blocktables_s + block_idx * stride_blocktables_b
    block_number = tl.load(block_table_ptr)
    
    # Base pointers for current key/value
    k_ptr = K + pid_seq * stride_k_b + pid_head * stride_k_h
    v_ptr = V + pid_seq * stride_v_b + pid_head * stride_v_h
    
    # Load all elements in head_dim dimension
    d_offsets = tl.arange(0, head_dim)
    k_vals = tl.load(k_ptr + d_offsets * stride_k_d)
    v_vals = tl.load(v_ptr + d_offsets * stride_v_d)
    
    # Compute cache offsets based on layout
    if new_cache_layout:
        # 5D layout: [b, h, d0, bs, x]
        d0 = d_offsets // x
        dx = d_offsets % x
        kcache_offsets = (block_number * stride_kcache_b +
                          pid_head * stride_kcache_h +
                          d0 * stride_kcache_d0 +
                          pos_in_block * stride_kcache_bs +
                          dx * stride_kcache_x)
        vcache_offsets = (block_number * stride_vcache_b +
                          pid_head * stride_vcache_h +
                          d0 * stride_vcache_d0 +
                          pos_in_block * stride_vcache_bs +
                          dx * stride_vcache_x)
    else:
        # 4D layout: [b, h, bs, d]
        kcache_offsets = (block_number * stride_kcache_b +
                          pid_head * stride_kcache_h +
                          pos_in_block * stride_kcache_bs +
                          d_offsets * stride_kcache_d)
        vcache_offsets = (block_number * stride_vcache_b +
                          pid_head * stride_vcache_h +
                          pos_in_block * stride_vcache_bs +
                          d_offsets * stride_vcache_d)
    
    # Store to cache
    tl.store(KCache + kcache_offsets, k_vals)
    tl.store(VCache + vcache_offsets, v_vals)

def copy_kv_to_blocked_cache(
    K: torch.Tensor,
    V: torch.Tensor,
    KCache: torch.Tensor,
    VCache: torch.Tensor,
    BlockTables: torch.Tensor,
    ContextLengths: torch.Tensor,
    block_size: int,
    new_cache_layout: bool = False
):
    """
    Efficiently copy keys/values to blocked KV cache.
    
    Args:
        K: Input keys tensor [batch_size, num_kv_heads, head_dim]
        V: Input values tensor [batch_size, num_kv_heads, head_dim]
        KCache: Keys cache tensor 
            If new_cache_layout: [num_blocks, num_heads, head_dim//x, block_size, x]
            Else: [num_blocks, num_heads, block_size, head_dim]
        VCache: Values cache tensor (same layout as KCache)
        BlockTables: [batch_size, max_blocks_per_seq] int tensor
        ContextLengths: [batch_size] int tensor of current lengths
        block_size: Size of each cache block
        new_cache_layout: Whether to use 5D layout
    """
    # Validate input shapes
    assert K.shape == V.shape, "Keys/Values must have same shape"
    batch_size, num_kv_heads, head_dim = K.shape
    assert BlockTables.shape[0] == batch_size, "Block table dimension mismatch"
    assert ContextLengths.shape == (batch_size,), "Context lengths shape mismatch"
    
    # Validate cache layouts
    if new_cache_layout:
        x = KCache.size(-1)  # Last dimension of 5D cache
        assert KCache.dim() == 5, "KCache must be 5D for new layout"
        assert head_dim % x == 0, f"Head dim {head_dim} must divide by {x}"
    else:
        assert KCache.dim() == 4, "KCache must be 4D for old layout"
        assert KCache.size(3) == head_dim, "Head dim mismatch in 4D cache"
    
    # Get strides for K/V
    stride_k_b, stride_k_h, stride_k_d = K.stride()
    stride_v_b, stride_v_h, stride_v_d = V.stride()
    
    # Get cache strides
    if new_cache_layout:
        s_kb, s_kh, s_kd0, s_kbs, s_kx = KCache.stride()
        s_vb, s_vh, s_vd0, s_vbs, s_vx = VCache.stride()
        # Dummy strides for 4D
        s_kd4 = s_vd4 = 0
    else:
        s_kb, s_kh, s_kbs, s_kd4 = KCache.stride()
        s_vb, s_vh, s_vbs, s_vd4 = VCache.stride()
        # Dummy strides for 5D
        s_kd0 = s_kx = s_vd0 = s_vx = 0
    
    # Block tables strides
    s_block_s, s_block_b = BlockTables.stride()
    
    # Launch kernel with 2D grid
    grid = (batch_size, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        K, V, KCache, VCache, BlockTables, ContextLengths,
        # K/V strides
        stride_k_b, stride_k_h, stride_k_d,
        stride_v_b, stride_v_h, stride_v_d,
        # KCache strides
        s_kb, s_kh, s_kbs, s_kd4, s_kd0, s_kx,
        # VCache strides
        s_vb, s_vh, s_vbs, s_vd4, s_vd0, s_vx,
        # Block tables strides
        s_block_s, s_block_b,
        # Parameters
        head_dim, block_size, new_cache_layout,
        x if new_cache_layout else 0  # x only used in new layout
    )
