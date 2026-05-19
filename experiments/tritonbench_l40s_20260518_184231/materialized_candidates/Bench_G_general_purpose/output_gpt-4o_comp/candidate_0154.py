import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q_ptr, k_ptr, v_ptr, k_cache_ptr, v_cache_ptr,
    sin_ptr, cos_ptr,
    q_stride, k_stride, v_stride, cache_stride,
    sin_stride, cos_stride,
    head_dim, kv_group_num, use_new_kcache_layout,
    n_heads, n_tokens, head_offset, token_offset,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    head_id = pid // n_tokens
    token_id = pid % n_tokens

    # Calculate indices
    q_idx = head_id * q_stride + token_id * head_dim
    k_idx = head_id * k_stride + token_id * head_dim
    v_idx = head_id * v_stride + token_id * head_dim

    # Load q, k, v
    q = tl.load(q_ptr + q_idx + head_offset)
    k = tl.load(k_ptr + k_idx + head_offset)
    v = tl.load(v_ptr + v_idx + head_offset)

    # Load sine and cosine values
    sin_idx = head_id * sin_stride
    cos_idx = head_id * cos_stride
    sin = tl.load(sin_ptr + sin_idx)
    cos = tl.load(cos_ptr + cos_idx)

    # Compute rotary embeddings
    q_rot = q * cos - tl.sin(q) * sin
    k_rot = k * cos - tl.sin(k) * sin

    # Store results back in q
    tl.store(q_ptr + q_idx + head_offset, q_rot)

    # Optionally update k_cache and v_cache
    if kv_group_num > 0:
        cache_idx = head_id * cache_stride + token_id * head_dim
        if use_new_kcache_layout:
            tl.store(k_cache_ptr + cache_idx, k_rot)
            tl.store(v_cache_ptr + cache_idx, v)
        else:
            # Implement old cache layout logic if needed
            pass

def decoding_fused_rotary_embedding(
    q, k, v, k_cache, v_cache, sin, cos,
    q_stride, k_stride, v_stride, cache_stride,
    sin_stride, cos_stride,
    head_dim, kv_group_num, use_new_kcache_layout,
    n_heads, n_tokens, block_tables, kv_lengths
):
    # Determine grid size
    grid = (n_heads * n_tokens,)

    # Choose number of warps based on head_dim
    num_warps = min(4, (head_dim + 31) // 32)

    # Launch kernel
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, k_cache, v_cache,
        sin, cos,
        q_stride, k_stride, v_stride, cache_stride,
        sin_stride, cos_stride,
        head_dim, kv_group_num, use_new_kcache_layout,
        n_heads, n_tokens, block_tables, kv_lengths,
        BLOCK_SIZE=128,
        num_warps=num_warps
    )
