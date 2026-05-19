import triton
import triton.language as tl
import torch

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    # Pointers to input and output tensors
    k_ptr, cache_ptr,
    # Dimensions and strides
    batch_size, num_heads, head_dim, seqlen,
    k_batch_stride, k_head_stride, k_seq_stride,
    cache_batch_stride, cache_head_stride, cache_seq_stride,
    # Cache block parameters
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
    new_cache_format: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and head indices
    batch_id = pid // num_heads
    head_id = pid % num_heads
    
    # Calculate offsets for input and cache
    k_offset = batch_id * k_batch_stride + head_id * k_head_stride
    cache_offset = batch_id * cache_batch_stride + head_id * cache_head_stride
    
    # Handle different cache formats
    if new_cache_format:
        # New format: blocked layout
        block_id = seqlen // BLOCK_K
        seq_in_block = seqlen % BLOCK_K
        cache_offset += (block_id * BLOCK_K + seq_in_block) * cache_seq_stride
    else:
        # Traditional format
        cache_offset += seqlen * cache_seq_stride
    
    # Load and store data in blocks
    for idx in range(0, head_dim, KCACHE_X):
        # Create block pointers
        k_block_ptr = k_ptr + k_offset + idx
        cache_block_ptr = cache_ptr + cache_offset + idx
        
        # Define block size
        block_size = min(KCACHE_X, head_dim - idx)
        
        # Load block from input
        k_block = tl.load(k_block_ptr, mask=idx < head_dim, other=0.0)
        
        # Store block to cache
        tl.store(cache_block_ptr, k_block, mask=idx < head_dim)

def copy_k_to_blocked_cache(k: torch.Tensor, cache: torch.Tensor, seqlen: int, new_cache_format: bool = True):
    """
    Copy key/value tensor to blocked cache format.
    
    Args:
        k: Input tensor of shape (batch_size, num_heads, seqlen, head_dim)
        cache: Cache tensor
        seqlen: Current sequence length
        new_cache_format: Whether to use new blocked cache format
    """
    batch_size, num_heads, _, head_dim = k.shape
    
    # Constants for cache blocking
    HEAD_DIM = 128
    BLOCK_K = 16  # Cache block size
    KCACHE_X = 32 # Vector load/store width
    
    # Verify dimensions
    assert head_dim <= HEAD_DIM, f"Head dimension {head_dim} exceeds maximum {HEAD_DIM}"
    
    # Calculate strides
    k_batch_stride = k.stride(0)
    k_head_stride = k.stride(1)
    k_seq_stride = k.stride(2)
    
    cache_batch_stride = cache.stride(0)
    cache_head_stride = cache.stride(1)
    cache_seq_stride = cache.stride(2)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _copy_to_kcache_seqlen_n_kernel[grid](
        k.data_ptr(), cache.data_ptr(),
        batch_size, num_heads, head_dim, seqlen,
        k_batch_stride, k_head_stride, k_seq_stride,
        cache_batch_stride, cache_head_stride, cache_seq_stride,
        BLOCK_K=BLOCK_K,
        HEAD_DIM=HEAD_DIM,
        KCACHE_X=KCACHE_X,
        new_cache_format=new_cache_format,
    )
