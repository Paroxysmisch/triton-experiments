import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    # Pointers to tensors
    k_ptr,
    cache_ptr,
    past_lengths_ptr,
    # Tensor strides
    k_stride_batch,
    k_stride_head,
    k_stride_seq,
    k_stride_dim,
    cache_stride_batch,
    cache_stride_head,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_dim,
    # Constants
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    cache_format: tl.constexpr,
    KCACHE_X: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Determine program IDs
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    # Check batch and head boundaries
    if pid_bh >= batch_size * num_heads:
        return
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    if batch_idx >= batch_size:
        return
    
    # Check sequence boundary
    seq_idx = pid_s
    if seq_idx >= seq_len:
        return
    
    # Calculate dimension indices with masking
    dim_idx = pid_d * KCACHE_X + tl.arange(0, KCACHE_X)
    mask = dim_idx < head_dim
    dim_idx = tl.max_contiguous(mask, KCACHE_X)
    
    # Load past length for current batch
    past_length = tl.load(past_lengths_ptr + batch_idx)
    current_pos = past_length + seq_idx
    
    # Calculate cache offset based on format
    if cache_format == 1:  # Blocked format
        block_idx = current_pos // BLOCK_SIZE
        block_offset = current_pos % BLOCK_SIZE
        cache_offset = (
            batch_idx * cache_stride_batch +
            head_idx * cache_stride_head +
            block_idx * cache_stride_block +
            block_offset * cache_stride_pos
        )
    else:  # Traditional format
        cache_offset = (
            batch_idx * cache_stride_batch +
            head_idx * cache_stride_head +
            current_pos * cache_stride_pos
        )
    
    # Compute final cache pointer
    cache_ptr_i = cache_ptr + cache_offset + dim_idx * cache_stride_dim
    
    # Compute key pointer
    k_offset = (
        batch_idx * k_stride_batch +
        head_idx * k_stride_head +
        seq_idx * k_stride_seq +
        dim_idx * k_stride_dim
    )
    k_ptr_i = k_ptr + k_offset
    
    # Load and store data
    k_val = tl.load(k_ptr_i, mask=mask, other=0)
    tl.store(cache_ptr_i, k_val, mask=mask)

import torch

def copy_k_to_blocked_cache(
    k: torch.Tensor,
    cache: torch.Tensor,
    past_lengths: torch.Tensor,
    cache_format: int,
    block_size: int = 128,
):
    # Validate input dimensions
    batch, num_heads, seq_len, head_dim = k.shape
    assert head_dim == cache.shape[-1], "Head dimension mismatch"
    
    # Check cache capacity based on format
    max_past_length = past_lengths.max().item()
    if cache_format == 1:
        assert len(cache.shape) == 5, "Blocked cache requires 5D shape"
        cache_blocks = cache.shape[2] * cache.shape[3]
        required_blocks = (max_past_length + seq_len + block_size - 1) // block_size
        assert cache_blocks >= required_blocks, "Insufficient cache blocks"
    else:
        assert len(cache.shape) == 4, "Traditional cache requires 4D shape"
        assert cache.shape[2] >= max_past_length + seq_len, "Insufficient cache length"
    
    # Ensure contiguous tensors
    k = k.contiguous()
    cache = cache.contiguous()
    past_lengths = past_lengths.contiguous()
    
    # Calculate strides for k
    k_stride_batch, k_stride_head, k_stride_seq, k_stride_dim = k.stride()
    
    # Calculate cache strides based on format
    if cache_format == 1:
        (cache_stride_batch, cache_stride_head, 
         cache_stride_block, cache_stride_pos, cache_stride_dim) = cache.stride()
    else:
        cache_stride_batch, cache_stride_head, cache_stride_pos, cache_stride_dim = cache.stride()
        cache_stride_block = 0  # Unused
    
    # Kernel configuration
    KCACHE_X = 16  # Process 16 elements per kernel instance
    grid = (
        batch * num_heads,  # Combine batch and heads for grid X
        seq_len,            # Sequence dimension for grid Y
        triton.cdiv(head_dim, KCACHE_X)  # Split head_dim into chunks
    )
    
    # Launch kernel
    _copy_to_kcache_seqlen_n_kernel[grid](
        k, cache, past_lengths,
        k_stride_batch, k_stride_head, k_stride_seq, k_stride_dim,
        cache_stride_batch, cache_stride_head, cache_stride_block, 
        cache_stride_pos, cache_stride_dim,
        seq_len, head_dim, block_size,
        batch, num_heads, cache_format,
        KCACHE_X=KCACHE_X,
        BLOCK_SIZE=block_size,
    )
