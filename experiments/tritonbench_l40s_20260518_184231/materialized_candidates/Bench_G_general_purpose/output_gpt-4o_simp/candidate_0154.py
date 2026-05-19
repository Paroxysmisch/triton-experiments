import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q_ptr, k_ptr, v_ptr,
    cos_ptr, sin_ptr,
    k_cache_ptr, v_cache_ptr,
    head_idx, seq_len, n_heads, d_head,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate block and thread indices
    block_idx = tl.program_id(0)
    thread_idx = tl.arange(0, BLOCK_SIZE)

    # Compute the offsets for q, k, v based on the block index
    q_offset = block_idx * seq_len * d_head + thread_idx
    k_offset = block_idx * seq_len * d_head + thread_idx
    v_offset = block_idx * seq_len * d_head + thread_idx

    # Load q, k, v
    q = tl.load(q_ptr + q_offset)
    k = tl.load(k_ptr + k_offset)
    v = tl.load(v_ptr + v_offset)

    # Load cos and sin values
    cos = tl.load(cos_ptr + thread_idx)
    sin = tl.load(sin_ptr + thread_idx)

    # Apply rotary embedding transformation
    q_rot = q * cos - k * sin
    k_rot = k * cos + q * sin

    # Store the results in the cache
    tl.store(k_cache_ptr + k_offset, k_rot)
    tl.store(v_cache_ptr + v_offset, v)

# Wrapper function
def decoding_fused_rotary_embedding(q, k, v, cos, sin, k_cache, v_cache, head_idx, seq_len, n_heads, d_head):
    # Calculate grid size
    grid_size = (n_heads, )

    # Launch the Triton kernel
    decoding_fused_rotary_embedding_kernel[grid_size](
        q, k, v,
        cos, sin,
        k_cache, v_cache,
        head_idx, seq_len, n_heads, d_head,
        BLOCK_SIZE=128  # Example block size
    )
