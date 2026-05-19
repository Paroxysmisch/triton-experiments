import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K, V,                    # Input key and value tensors
    KCache, VCache,          # Cache tensors to store K/V
    BLOCK_TABLES,           # Block mapping information
    context_lengths,        # Past sequence lengths
    # Strides for input tensors
    stride_kt, stride_kh, stride_kd,  # Key tensor strides
    stride_vt, stride_vh, stride_vd,  # Value tensor strides
    # Strides for cache tensors
    stride_kcb, stride_kch, stride_kcsplit_x, stride_kcs, stride_kcd,  # KCache strides
    stride_vcb, stride_vch, stride_vcs, stride_vcd,                    # VCache strides
    # Block table strides
    stride_bts, stride_btb,
    block_size,             # Size of each cache block
    HEAD_DIM: tl.constexpr, # Head dimension size
    KCACHE_X: tl.constexpr, # Cache layout parameter
):
    # Get current sequence and head indices
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    # Calculate position in the cache
    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    # Initialize offset arrays
    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    # Copy data in chunks of size KCACHE_X
    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        # Calculate offsets for current chunk
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        
        # Load key and value data
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        # Store in cache with appropriate offsets
        offsets_kcache = (
            block_id * stride_kcb +
            cur_kv_head_idx * stride_kch +
            split_x * stride_kcsplit_x +
            offsets_in_last_block * stride_kcs +
            range_x
        )
        offsets_vcache = (
            block_id * stride_vcb +
            cur_kv_head_idx * stride_vch +
            offsets_in_last_block * stride_vcs +
            offsets_dmodel_x_partition * stride_vcd
        )
        
        # Store data in cache
        tl.store(KCache + offsets_kcache, k)
        tl.store(VCache + offsets_vcache, v)

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
    Copy keys and values to blocked cache during decoding.
    
    Args:
        k: Key tensor [bsz, 1, num_kv_heads, head_dim] or [bsz, num_kv_heads, head_dim]
        v: Value tensor [bsz, 1, num_kv_heads, head_dim] or [bsz, num_kv_heads, head_dim]
        k_cache: Key cache [num_blocks, num_kv_heads, block_size, head_dim] or
                [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        v_cache: Value cache with same layout as k_cache
        kv_lengths: Past sequence lengths [bsz]
        block_tables: Block mapping info [bsz, max_blocks_per_sequence]
        use_new_kcache_layout: Whether to use 5D cache layout
    """
    # Input validation and shape handling
    k = k.squeeze(1) if k.dim() == 4 else k
    v = v.squeeze(1) if v.dim() == 4 else v
    assert k.dim() == 3 and v.dim() == 3, "Invalid input dimensions"
    
    bsz, num_kv_heads, head_dim = k.shape
    block_size = k_cache.size(-2)

    # Handle different cache layouts
    x = head_dim
    stride_kcsplit_x, stride_kcs, stride_kcd = 0, k_cache.stride(2), k_cache.stride(3)
    if use_new_kcache_layout:
        x = k_cache.size(-1)
        stride_kcsplit_x, stride_kcs, stride_kcd = k_cache.stride()[2:]

    # Launch kernel
    grid = (bsz, num_kv_heads)
    num_warps = 8 if head_dim > 128 else 4
    
    _copy_to_kvcache_seqlen1_kernel[grid](
        k, v, k_cache, v_cache, block_tables, kv_lengths,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_cache.stride(0), k_cache.stride(1),
        stride_kcsplit_x, stride_kcs, stride_kcd,
        v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        block_size, HEAD_DIM=head_dim, KCACHE_X=x,
        num_warps=num_warps,
    )
