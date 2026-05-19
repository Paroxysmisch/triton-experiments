import triton
import triton.language as tl
import torch

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    # Pointers to input tensors
    K_ptr, V_ptr,
    # Pointers to output cache tensors
    KCache_ptr, VCache_ptr,
    # Dimensions and strides
    head_dim, block_size,
    K_batch_stride, K_head_stride,
    KCache_batch_stride, KCache_head_stride, KCache_seq_stride,
    # Position information
    cache_offset,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for batch and head dimensions
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate base pointers for current batch and head
    k_base_ptr = K_ptr + batch_idx * K_batch_stride + head_idx * K_head_stride
    v_base_ptr = V_ptr + batch_idx * K_batch_stride + head_idx * K_head_stride
    
    # Calculate cache positions
    cache_base_idx = batch_idx * KCache_batch_stride + head_idx * KCache_head_stride
    seq_idx = cache_offset
    cache_seq_offset = seq_idx * KCache_seq_stride
    
    # Load and store data in blocks
    for block_start in range(0, head_dim, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, head_dim)
        block_size = block_end - block_start
        
        # Create offset array for current block
        offs = block_start + tl.arange(0, block_size)
        
        # Load key and value data
        k_vals = tl.load(k_base_ptr + offs)
        v_vals = tl.load(v_base_ptr + offs)
        
        # Store to cache
        k_cache_ptr = KCache_ptr + cache_base_idx + cache_seq_offset + offs
        v_cache_ptr = VCache_ptr + cache_base_idx + cache_seq_offset + offs
        
        tl.store(k_cache_ptr, k_vals)
        tl.store(v_cache_ptr, v_vals)

def copy_kv_to_blocked_cache(K: torch.Tensor, V: torch.Tensor, 
                           KCache: torch.Tensor, VCache: torch.Tensor,
                           cache_offset: int):
    """
    Copy K/V tensors to blocked format cache.
    
    Args:
        K: Key tensor of shape (batch_size, num_kv_heads, head_dim)
        V: Value tensor of shape (batch_size, num_kv_heads, head_dim) 
        KCache: Key cache tensor
        VCache: Value cache tensor
        cache_offset: Current position in the cache
    """
    batch_size, num_kv_heads, head_dim = K.shape
    
    # Configure block size for the kernel
    BLOCK_SIZE = 128
    
    # Launch kernel with grid dimensions
    grid = (batch_size, num_kv_heads)
    
    # Get strides for tensors
    K_batch_stride = K.stride(0)
    K_head_stride = K.stride(1)
    KCache_batch_stride = KCache.stride(0)
    KCache_head_stride = KCache.stride(1)
    KCache_seq_stride = KCache.stride(2)
    
    # Launch Triton kernel
    _copy_to_kvcache_seqlen1_kernel[grid](
        K, V,
        KCache, VCache,
        head_dim, BLOCK_SIZE,
        K_batch_stride, K_head_stride,
        KCache_batch_stride, KCache_head_stride, KCache_seq_stride,
        cache_offset,
        BLOCK_SIZE=BLOCK_SIZE,
    )
