import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    block_tables_ptr, kv_lengths_ptr,
    # Tensor strides and offsets
    q_stride_s, q_stride_h, q_stride_d,
    k_stride_s, k_stride_h, k_stride_d,
    v_stride_s, v_stride_h, v_stride_d,
    k_cache_stride_b, k_cache_stride_s, k_cache_stride_h, k_cache_stride_d,
    v_cache_stride_b, v_cache_stride_s, v_cache_stride_h, v_cache_stride_d,
    block_tables_stride_b, block_tables_stride_n,
    # Parameters
    KV_GROUP_NUM: tl.constexpr,
    use_new_kcache_layout: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # Other constants
    BLOCK_TOKENS: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_token = tl.program_id(1)

    batch_id = pid_token
    seq_pos = tl.load(kv_lengths_ptr + batch_id)
    pos = seq_pos

    # Calculate block index and offset in cache
    block_idx = pos // BLOCK_SIZE
    block_offset = pos % BLOCK_SIZE

    # Load physical block from block tables
    block_table_offset = batch_id * block_tables_stride_b + block_idx * block_tables_stride_n
    physical_block = tl.load(block_tables_ptr + block_table_offset)

    # Compute KV head index
    kv_head_idx = pid_head // KV_GROUP_NUM

    # Compute base offsets for q, k, v
    q_offset = batch_id * q_stride_s + pid_head * q_stride_h
    k_offset = batch_id * k_stride_s + pid_head * k_stride_h
    v_offset = batch_id * v_stride_s + pid_head * v_stride_h

    # Compute cache offsets
    if use_new_kcache_layout:
        k_cache_offset = (physical_block * k_cache_stride_b +
                           kv_head_idx * k_cache_stride_h +
                           block_offset * k_cache_stride_s)
        v_cache_offset = (physical_block * v_cache_stride_b +
                          kv_head_idx * v_cache_stride_h +
                          block_offset * v_cache_stride_s)
    else:
        k_cache_offset = (physical_block * k_cache_stride_b +
                           block_offset * k_cache_stride_s +
                           kv_head_idx * k_cache_stride_h)
        v_cache_offset = (physical_block * v_cache_stride_b +
                           block_offset * v_cache_stride_s +
                           kv_head_idx * v_cache_stride_h)

    half_dim = head_dim // 2
    for i in tl.static_range(half_dim):
        # Rotary embedding for Q
        q_i = tl.load(q_ptr + q_offset + i * q_stride_d)
        q_j = tl.load(q_ptr + q_offset + (i + half_dim) * q_stride_d)

        # Compute frequency
        inv_freq = 1.0 / (10000 ** (2 * i / head_dim))
        freq = pos * inv_freq
        cos = tl.cos(freq)
        sin = tl.sin(freq)

        # Apply rotation
        q_i_rot = q_i * cos - q_j * sin
        q_j_rot = q_i * sin + q_j * cos

        tl.store(q_ptr + q_offset + i * q_stride_d, q_i_rot)
        tl.store(q_ptr + q_offset + (i + half_dim) * q_stride_d, q_j_rot)

        # Rotary embedding for K and V, then store to cache
        k_i = tl.load(k_ptr + k_offset + i * k_stride_d)
        k_j = tl.load(k_ptr + k_offset + (i + half_dim) * k_stride_d)
        k_i_rot = k_i * cos - k_j * sin
        k_j_rot = k_i * sin + k_j * cos
        tl.store(k_cache_ptr + k_cache_offset + i * k_cache_stride_d, k_i_rot)
        tl.store(k_cache_ptr + k_cache_offset + (i + half_dim) * k_cache_stride_d, k_j_rot)

        v_i = tl.load(v_ptr + v_offset + i * v_stride_d)
        v_j = tl.load(v_ptr + v_offset + (i + half_dim) * v_stride_d)
        v_i_rot = v_i * cos - v_j * sin
        v_j_rot = v_i * sin + v_j * cos
        tl.store(v_cache_ptr + v_cache_offset + i * v_cache_stride_d, v_i_rot)
        tl.store(v_cache_ptr + v_cache_offset + (i + half_dim) * v_cache_stride_d, v_j_rot)

def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lengths: torch.Tensor,
    KV_GROUP_NUM: int,
    use_new_kcache_layout: bool,
    BLOCK_SIZE: int,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert k_cache.is_cuda and v_cache.is_cuda
    assert block_tables.is_cuda and kv_lengths.is_cuda

    batch_size, q_num_heads, head_dim = q.shape
    total_tokens = batch_size  # One token per batch in decoding

    num_warps = 4 if head_dim <= 64 else 8

    # Strides for Q, K, V
    q_stride_s, q_stride_h, q_stride_d = q.stride()
    k_stride_s, k_stride_h, k_stride_d = k.stride()
    v_stride_s, v_stride_h, v_stride_d = v.stride()

    # Cache strides
    if use_new_kcache_layout:
        k_cache_stride_b, k_cache_stride_h, k_cache_stride_s, k_cache_stride_d = k_cache.stride()
        v_cache_stride_b, v_cache_stride_h, v_cache_stride_s, v_cache_stride_d = v_cache.stride()
    else:
        k_cache_stride_b, k_cache_stride_s, k_cache_stride_h, k_cache_stride_d = k_cache.stride()
        v_cache_stride_b, v_cache_stride_s, v_cache_stride_h, v_cache_stride_d = v_cache.stride()

    # Block tables strides
    block_tables_stride_b, block_tables_stride_n = block_tables.stride()

    grid = (q_num_heads, total_tokens)
    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, k_cache, v_cache,
        block_tables, kv_lengths,
        q_stride_s, q_stride_h, q_stride_d,
        k_stride_s, k_stride_h, k_stride_d,
        v_stride_s, v_stride_h, v_stride_d,
        k_cache_stride_b, k_cache_stride_s, k_cache_stride_h, k_cache_stride_d,
        v_cache_stride_b, v_cache_stride_s, v_cache_stride_h, v_cache_stride_d,
        block_tables_stride_b, block_tables_stride_n,
        KV_GROUP_NUM,
        use_new_kcache_layout,
        head_dim,
        BLOCK_SIZE,
        BLOCK_TOKENS=1,
        num_warps=num_warps,
    )
