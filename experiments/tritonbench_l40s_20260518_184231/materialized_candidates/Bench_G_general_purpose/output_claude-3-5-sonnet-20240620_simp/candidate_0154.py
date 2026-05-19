import triton
import triton.language as tl
import torch

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr,                    # Input tensors
    cos_ptr, sin_ptr,                       # Rotary embedding coefficients
    k_cache_ptr, v_cache_ptr,               # Cache tensors
    # Dimensions and strides
    head_dim, num_heads,                    # Dimensions
    q_stride_b, q_stride_h, q_stride_d,     # Query strides
    k_stride_b, k_stride_h, k_stride_d,     # Key strides
    v_stride_b, v_stride_h, v_stride_d,     # Value strides
    cos_stride_t, cos_stride_d,             # Cosine strides
    sin_stride_t, sin_stride_d,             # Sine strides
    k_cache_stride_b, k_cache_stride_h,     # Key cache strides
    k_cache_stride_t, k_cache_stride_d,
    v_cache_stride_b, v_cache_stride_h,     # Value cache strides
    v_cache_stride_t, v_cache_stride_d,
    # Other parameters
    timestep, batch_size,                   # Current timestep and batch size
    BLOCK_SIZE: tl.constexpr):
    
    # Calculate indices
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    
    # Handle case where we've exceeded batch * heads
    if batch_idx >= batch_size:
        return
        
    # Calculate base pointers for this batch and head
    q_base = q_ptr + batch_idx * q_stride_b + head_idx * q_stride_h
    k_base = k_ptr + batch_idx * k_stride_b + head_idx * k_stride_h
    v_base = v_ptr + batch_idx * v_stride_b + head_idx * v_stride_h
    
    # Cache base pointers
    k_cache_base = k_cache_ptr + batch_idx * k_cache_stride_b + head_idx * k_cache_stride_h + timestep * k_cache_stride_t
    v_cache_base = v_cache_ptr + batch_idx * v_cache_stride_b + head_idx * v_cache_stride_h + timestep * v_cache_stride_t
    
    # Load offsets for rotary embedding
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Load query, key, and value
    q = tl.load(q_base + offs * q_stride_d, mask=offs < head_dim)
    k = tl.load(k_base + offs * k_stride_d, mask=offs < head_dim)
    v = tl.load(v_base + offs * v_stride_d, mask=offs < head_dim)
    
    # Load rotary embedding coefficients
    cos = tl.load(cos_ptr + timestep * cos_stride_t + offs * cos_stride_d, mask=offs < head_dim)
    sin = tl.load(sin_ptr + timestep * sin_stride_t + offs * sin_stride_d, mask=offs < head_dim)
    
    # Apply rotary embeddings
    q_rot = q * cos + tl.roll(q, 1) * sin
    k_rot = k * cos + tl.roll(k, 1) * sin
    
    # Store to cache
    tl.store(k_cache_base + offs * k_cache_stride_d, k_rot, mask=offs < head_dim)
    tl.store(v_cache_base + offs * v_cache_stride_d, v, mask=offs < head_dim)
    
    # Store rotated query and key back
    tl.store(q_base + offs * q_stride_d, q_rot, mask=offs < head_dim)
    tl.store(k_base + offs * k_stride_d, k_rot, mask=offs < head_dim)

def decoding_fused_rotary_embedding(q, k, v, cos, sin, k_cache, v_cache, timestep):
    """
    Wrapper function for the fused rotary embedding kernel
    
    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k: Key tensor [batch_size, num_heads, head_dim]
        v: Value tensor [batch_size, num_heads, head_dim]
        cos: Cosine tensor [max_seq_len, head_dim]
        sin: Sine tensor [max_seq_len, head_dim]
        k_cache: Key cache tensor [batch_size, num_heads, max_seq_len, head_dim]
        v_cache: Value cache tensor [batch_size, num_heads, max_seq_len, head_dim]
        timestep: Current timestep
    """
    batch_size, num_heads, head_dim = q.shape
    
    # Calculate grid size
    grid = (batch_size * num_heads,)
    
    # Determine optimal block size (must be power of 2)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    # Launch kernel
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v,
        cos, sin,
        k_cache, v_cache,
        head_dim, num_heads,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), v_cache.stride(3),
        timestep, batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return q, k, v
