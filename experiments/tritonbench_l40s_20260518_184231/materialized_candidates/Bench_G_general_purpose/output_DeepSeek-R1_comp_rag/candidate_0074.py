import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K,
    V,
    KCache,
    VCache,
    BLOCK_TABLES,
    context_lengths,
    stride_kt,  # token stride for K
    stride_kh,  # head stride for K
    stride_kd,  # dim stride for K
    stride_vt,  # token stride for V
    stride_vh,  # head stride for V
    stride_vd,  # dim stride for V
    stride_kcb, # block stride for KCache
    stride_kch, # head stride for KCache
    stride_kcx, # split x stride for KCache (for 5D layout)
    stride_kcs, # sequence stride for KCache
    stride_kcd, # dim stride for KCache
    stride_vcb, # block stride for VCache
    stride_vch, # head stride for VCache
    stride_vcs, # sequence stride for VCache
    stride_vcd, # dim stride for VCache
    stride_bts, # sequence stride for block tables
    stride_btb, # block stride for block tables
    block_size: int,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    # Get current sequence and head index
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)
    
    # Calculate position in cache
    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    offset_in_block = past_kv_seq_len % block_size
    
    # Get block table entry for this sequence
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    
    # Load K and V values
    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        # K Cache indexing
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + split_x * KCACHE_X + tl.arange(0, KCACHE_X)
        k_val = tl.load(K + offsets_k)
        
        # Cache indices for 4D or 5D layout
        kcache_offset = (
            block_id * stride_kcb +
            cur_kv_head_idx * stride_kch +
            split_x * stride_kcx * (KCACHE_X > 1) +  # only for 5D layout
            offset_in_block * stride_kcs +
            tl.arange(0, KCACHE_X)
        )
        tl.store(KCache + kcache_offset, k_val)
        
        # V Cache (always 4D layout)
        v_offsets = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + split_x * KCACHE_X + tl.arange(0, KCACHE_X)
        v_val = tl.load(V + v_offsets)
        vcache_offset = (
            block_id * stride_vcb +
            cur_kv_head_idx * stride_vch +
            offset_in_block * stride_vcs + 
            (split_x * KCACHE_X + tl.arange(0, KCACHE_X)) * stride_vcd
        )
        tl.store(VCache + vcache_offset, v_val)

def copy_kv_to_blocked_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    use_new_kcache_layout: bool = False,
):
    """
    Copies key and value tensors into blocked cache during decoding.
    
    Args:
        k: Input keys tensor [batch_size, num_heads, head_dim]
        v: Input values tensor [batch_size, num_heads, head_dim]
        k_cache: Blocked key cache tensor 
            Shape: [num_blocks, num_heads, block_size, head_dim] (4D) or 
                   [num_blocks, num_heads, split_dim, block_size, x] (5D)
        v_cache: Blocked value cache tensor [num_blocks, num_heads, block_size, head_dim]
        kv_lengths: Tensor of current sequence lengths [batch_size]
        block_tables: Block allocation table [batch_size, max_blocks_per_sequence]
        use_new_kcache_layout: Whether K cache uses 5D layout
    """
    # Validate inputs
    assert k.dim() == 3 and v.dim() == 3, "Input tensors must be 3D"
    batch_size, num_heads, head_dim = k.shape
    
    # Determine cache parameters
    kcache_shape = k_cache.shape
    block_size = k_cache.shape[-2] if use_new_kcache_layout else k_cache.shape[2]
    x_dim = k_cache.size(-1) if use_new_kcache_layout else head_dim
    
    # Setup kernel parameters
    grid = (batch_size, num_heads)
    num_warps = 8 if head_dim > 128 else 4
    
    # Determine K cache strides based on layout
    kstride_x = k_cache.stride(2) if use_new_kcache_layout else 0
    
    # Launch kernel
    _copy_to_kvcache_seqlen1_kernel[grid](
        K=k,
        V=v,
        KCache=k_cache,
        VCache=v_cache,
        BLOCK_TABLES=block_tables,
        context_lengths=kv
