import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    sin_ptr, cos_ptr,
    block_tables_ptr,
    kv_lengths_ptr,
    
    # Dimensions and strides
    head_dim: tl.constexpr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_t: tl.constexpr,
    kv_stride_b: tl.constexpr,
    kv_stride_h: tl.constexpr,
    kv_stride_t: tl.constexpr,
    kv_block_stride: tl.constexpr,
    kv_head_stride: tl.constexpr,
    
    # Other parameters
    rotary_dim: tl.constexpr,
    q_position_ids: tl.constexpr,
    kv_position_ids: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    use_new_kcache_layout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Program ID gives current head and token
    head_idx = tl.program_id(0)
    token_idx = tl.program_id(1)
    
    # Calculate offsets
    half_rotary_dim = rotary_dim // 2
    rotary_offset = tl.arange(0, BLOCK_SIZE)
    
    # Load position IDs and compute sin/cos indices
    pos_q = q_position_ids + token_idx
    sin_index = pos_q * rotary_dim + rotary_offset
    cos_index = sin_index
    
    # Load query values
    q_offset = head_idx * q_stride_h + token_idx * q_stride_t
    q = tl.load(q_ptr + q_offset + rotary_offset)
    
    # Load sin/cos values
    sin = tl.load(sin_ptr + sin_index)
    cos = tl.load(cos_ptr + cos_index)
    
    # Apply rotary embedding to first half
    q_rot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    q_pass = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Split into rotation and pass-through parts
    q_rot = q[:half_rotary_dim]
    q_pass = q[half_rotary_dim:]
    
    # Apply rotation
    q_rot_transformed = q_rot * cos[:half_rotary_dim] - tl.roll(q_rot, 1) * sin[:half_rotary_dim]
    
    # Store results back
    tl.store(q_ptr + q_offset, q_rot_transformed, mask=rotary_offset < half_rotary_dim)
    tl.store(q_ptr + q_offset + half_rotary_dim, q_pass, mask=rotary_offset < (rotary_dim - half_rotary_dim))
    
    # Update KV cache if needed
    if token_idx < KV_GROUP_NUM:
        # Calculate cache indices
        block_table_offset = token_idx if use_new_kcache_layout else 0
        block_idx = tl.load(block_tables_ptr + block_table_offset)
        
        # Load and transform K
        k_cache_offset = (block_idx * kv_block_stride + 
                         head_idx * kv_head_stride)
        k = tl.load(k_ptr + head_idx * kv_stride_h + token_idx * kv_stride_t + rotary_offset)
        k_transformed = k * cos - tl.roll(k, 1) * sin
        tl.store(k_cache_ptr + k_cache_offset, k_transformed)
        
        # Load and store V (no rotation needed)
        v_cache_offset = k_cache_offset  # Same layout as K
        v = tl.load(v_ptr + head_idx * kv_stride_h + token_idx * kv_stride_t + rotary_offset)
        tl.store(v_cache_ptr + v_cache_offset, v)

def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lengths: torch.Tensor,
    q_position_ids: int,
    kv_position_ids: int,
    KV_GROUP_NUM: int,
    use_new_kcache_layout: bool = False
):
    # Get dimensions
    batch_size, q_head_num, q_total_tokens, head_dim = q.shape
    rotary_dim = sin.shape[-1]
    
    # Calculate strides
    q_stride_b = q.stride(0)
    q_stride_h = q.stride(1)
    q_stride_t = q.stride(2)
    
    kv_stride_b = k.stride(0)
    kv_stride_h = k.stride(1)
    kv_stride_t = k.stride(2)
    
    if use_new_kcache_layout:
        kv_block_stride = k_cache.stride(1)
        kv_head_stride = k_cache.stride(2)
    else:
        kv_block_stride = k_cache.stride(0)
        kv_head_stride = k_cache.stride(1)
    
    # Choose number of warps based on head dimension
    num_warps = 4 if head_dim <= 64 else 8
    
    # Launch kernel
    grid = (q_head_num, q_total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v,
        k_cache, v_cache,
        sin, cos,
        block_tables,
        kv_lengths,
        head_dim=head_dim,
        q_stride_b=q_stride_b,
        q_stride_h=q_stride_h,
        q_stride_t=q_stride_t,
        kv_stride_b=kv_stride_b,
        kv_stride_h=kv_stride_h,
        kv_stride_t=kv_stride_t,
        kv_block_stride=kv_block_stride,
        kv_head_stride=kv_head_stride,
        rotary_dim=rotary_dim,
        q_position_ids=q_position_ids,
        kv_position_ids=kv_position_ids,
        KV_GROUP_NUM=KV_GROUP_NUM,
        use_new_kcache_layout=use_new_kcache_layout,
        num_warps=num_warps
    )
