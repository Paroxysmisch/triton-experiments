import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    K,                      # Key or Value tensor
    KCache,                 # Cache tensor
    BLOCK_TABLES,          # Block mapping table
    seq_lengths,           # Sequence lengths
    stride_kt,             # Stride for token dimension in K
    stride_kh,             # Stride for head dimension in K
    stride_kd,             # Stride for hidden dimension in K
    stride_kcb,           # Stride for block dimension in cache
    stride_kch,           # Stride for head dimension in cache
    stride_kcsplit_x,     # Stride for split dimension in cache
    stride_kcs,           # Stride for sequence dimension in cache
    stride_kcx,           # Stride for hidden dimension in cache
    stride_bts,           # Stride for sequence dimension in block table
    stride_btb,           # Stride for block dimension in block table
    block_size,           # Size of each block
    n_tokens,             # Number of tokens to copy
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    # Calculate indices
    pid = tl.program_id(0)
    cur_seq_idx = pid // n_tokens
    token_offset = pid % n_tokens
    
    # Get head and split indices
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    # Calculate position in sequence
    seq_len = tl.load(seq_lengths + cur_seq_idx)
    pos = seq_len + token_offset
    
    # Calculate block and offset
    block_idx = pos // block_size
    block_offset = pos % block_size
    
    # Get physical block ID from block table
    block_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts + block_idx * stride_btb
    physical_block = tl.load(block_ptr)

    # Calculate offsets for loading/storing
    offs_d = split_idx * KCACHE_X + tl.arange(0, KCACHE_X)
    
    # Load from input tensor
    offs_k = pid * stride_kt + head_idx * stride_kh + offs_d * stride_kd
    k_vals = tl.load(K + offs_k)
    
    # Store to cache
    offs_cache = (
        physical_block * stride_kcb +
        head_idx * stride_kch +
        split_idx * stride_kcsplit_x +
        block_offset * stride_kcs +
        tl.arange(0, KCACHE_X)
    )
    tl.store(KCache + offs_cache, k_vals)

def copy_k_to_blocked_cache(
    k: torch.Tensor,
    k_cache: torch.Tensor, 
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    n: int = 1,
    use_new_layout: bool = False
):
    """
    Copy key tensor to blocked cache format.
    
    Args:
        k: Input key tensor [batch_size, num_heads, head_dim] or [batch_size*n, num_heads, head_dim]
        k_cache: Cache tensor [num_blocks, num_heads, block_size, head_dim] or 
                [num_blocks, num_heads, head_dim//x, block_size, x] for new layout
        kv_lengths: Sequence lengths [batch_size]
        block_tables: Block mapping tables [batch_size, max_blocks]
        n: Number of tokens to copy per sequence
        use_new_layout: Whether to use new cache layout format
    """
    # Handle input shape
    if k.dim() == 4:
        k = k.reshape(-1, k.size(-2), k.size(-1))
    
    batch_size = k.size(0) // n
    num_heads = k.size(1)
    head_dim = k.size(2)
    
    # Get cache parameters
    block_size = k_cache.size(-2)
    x_factor = k_cache.size(-1) if use_new_layout else head_dim
    
    # Calculate strides
    if use_new_layout:
        stride_split, stride_seq, stride_dim = k_cache.stride()[2:5]
    else:
        stride_split = 0
        stride_seq = k_cache.stride(2)
        stride_dim = k_cache.stride(3)

    # Launch kernel
    grid = (batch_size * n, num_heads, head_dim // x_factor)
    num_warps = 8 if head_dim > 128 else 4
    
    _copy_to_kcache_seqlen_n_kernel[grid](
        k, k_cache, block_tables, kv_lengths,
        k.stride(0), k.stride(1), k.stride(2),
        k_cache.stride(0), k_cache.stride(1),
        stride_split, stride_seq, stride_dim,
        block_tables.stride(0), block_tables.stride(1),
        block_size, n,
        HEAD_DIM=head_dim,
        KCACHE_X=x_factor,
        num_warps=num_warps
    )
