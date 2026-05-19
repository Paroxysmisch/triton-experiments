import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    # Pointers to input tensors
    K_ptr, V_ptr,
    # Pointers to cache tensors  
    KCache_ptr, VCache_ptr,
    # Pointer to block tables
    BLOCK_TABLES_ptr,
    # Pointer to context lengths
    context_lengths_ptr,
    # Strides for input tensors
    stride_k_batch, stride_k_head, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_dim,
    # Strides for cache tensors
    stride_kc_block, stride_kc_head, stride_kc_seq, stride_kc_dim,
    stride_vc_block, stride_vc_head, stride_vc_seq, stride_vc_dim,
    # Strides for block tables
    stride_bt_batch, stride_bt_block,
    # Other parameters
    block_size: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Get indices from program ID
    batch_idx = tl.program_id(0)  # Batch dimension
    head_idx = tl.program_id(1)   # Head dimension
    
    # Calculate position in sequence
    seq_len = tl.load(context_lengths_ptr + batch_idx) - 1
    block_idx = seq_len // block_size
    seq_offset = seq_len % block_size
    
    # Load block ID from block tables
    block_table_offset = batch_idx * stride_bt_batch + block_idx * stride_bt_block
    block_id = tl.load(BLOCK_TABLES_ptr + block_table_offset)
    
    # Create offset arrays for the head dimension
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Calculate input offsets
    k_offset = batch_idx * stride_k_batch + head_idx * stride_k_head + offs_d * stride_k_dim
    v_offset = batch_idx * stride_v_batch + head_idx * stride_v_head + offs_d * stride_v_dim
    
    # Load input values
    k = tl.load(K_ptr + k_offset)
    v = tl.load(V_ptr + v_offset)
    
    # Calculate cache offsets
    kc_offset = (block_id * stride_kc_block + 
                head_idx * stride_kc_head + 
                seq_offset * stride_kc_seq + 
                offs_d * stride_kc_dim)
    
    vc_offset = (block_id * stride_vc_block + 
                head_idx * stride_vc_head + 
                seq_offset * stride_vc_seq + 
                offs_d * stride_vc_dim)
    
    # Store to cache
    tl.store(KCache_ptr + kc_offset, k)
    tl.store(VCache_ptr + vc_offset, v)

def copy_kv_to_blocked_cache(
    k: torch.Tensor,
    v: torch.Tensor, 
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
):
    """
    Copy K/V tensors to blocked cache format.
    
    Args:
        k: Key tensor [batch_size, num_heads, head_dim]
        v: Value tensor [batch_size, num_heads, head_dim] 
        k_cache: Key cache [num_blocks, num_heads, block_size, head_dim]
        v_cache: Value cache [num_blocks, num_heads, block_size, head_dim]
        context_lengths: Sequence lengths [batch_size]
        block_tables: Block mapping [batch_size, max_blocks]
    """
    # Get dimensions
    batch_size = k.shape[0]
    num_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = k_cache.shape[2]
    
    # Launch kernel
    grid = (batch_size, num_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        k, v,
        k_cache, v_cache, 
        block_tables,
        context_lengths,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        block_size=block_size,
        HEAD_DIM=head_dim,
        num_warps=4
    )
