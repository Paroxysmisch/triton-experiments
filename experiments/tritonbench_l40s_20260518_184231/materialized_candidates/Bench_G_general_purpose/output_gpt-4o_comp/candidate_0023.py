import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    out_q_ptr, out_k_ptr,
    q_stride, k_stride, cos_stride, sin_stride,
    batch_size, seq_len, head_num, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (seq_len * head_num)
    seq_id = (pid // head_num) % seq_len
    head_id = pid % head_num

    # Compute offsets
    q_offset = batch_id * q_stride[0] + seq_id * q_stride[1] + head_id * q_stride[2]
    k_offset = batch_id * k_stride[0] + seq_id * k_stride[1] + head_id * k_stride[2]
    cos_offset = head_id * cos_stride[0]
    sin_offset = head_id * sin_stride[0]

    # Load values
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE))
    k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_SIZE))
    cos = tl.load(cos_ptr + cos_offset + tl.arange(0, BLOCK_SIZE))
    sin = tl.load(sin_ptr + sin_offset + tl.arange(0, BLOCK_SIZE))

    # Apply rotary embeddings
    q_rotated = q * cos - tl.rotate(q, 1) * sin
    k_rotated = k * cos - tl.rotate(k, 1) * sin

    # Store results
    tl.store(out_q_ptr + q_offset + tl.arange(0, BLOCK_SIZE), q_rotated)
    tl.store(out_k_ptr + k_offset + tl.arange(0, BLOCK_SIZE), k_rotated)

# Wrapper function
def rotary_embedding(q, k, cos, sin, head_num, head_dim):
    batch_size, seq_len, _, _ = q.shape
    BLOCK_SIZE = head_dim

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    grid = (batch_size * seq_len * head_num,)
    rotary_embedding_kernel[grid](
        q, k, cos, sin, out_q, out_k,
        q.stride(), k.stride(), cos.stride(), sin.stride(),
        batch_size, seq_len, head_num, head_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_q, out_k

@triton.jit
def fused_rotary_embedding_kernel_v2(
    q_ptr, k_ptr, cos_ptr, sin_ptr, k_cache_ptr,
    block_tables_ptr, kv_lengths_ptr,
    out_q_ptr, out_k_ptr,
    q_stride, k_stride, cos_stride, sin_stride, k_cache_stride,
    block_tables_stride, kv_lengths_stride,
    batch_size, seq_len, head_num, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_id = pid // (seq_len * head_num)
    seq_id = (pid // head_num) % seq_len
    head_id = pid % head_num

    # Compute offsets
    q_offset = batch_id * q_stride[0] + seq_id * q_stride[1] + head_id * q_stride[2]
    k_offset = batch_id * k_stride[0] + seq_id * k_stride[1] + head_id * k_stride[2]
    cos_offset = head_id * cos_stride[0]
    sin_offset = head_id * sin_stride[0]

    # Load values
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE))
    cos = tl.load(cos_ptr + cos_offset + tl.arange(0, BLOCK_SIZE))
    sin = tl.load(sin_ptr + sin_offset + tl.arange(0, BLOCK_SIZE))

    # Apply rotary embeddings
    q_rotated = q * cos - tl.rotate(q, 1) * sin

    # Store results
    tl.store(out_q_ptr + q_offset + tl.arange(0, BLOCK_SIZE), q_rotated)

    # Handle k_cache
    if k_cache_ptr:
        block_table_offset = batch_id * block_tables_stride[0] + head_id * block_tables_stride[1]
        kv_length_offset = batch_id * kv_lengths_stride[0] + head_id * kv_lengths_stride[1]

        block_table = tl.load(block_tables_ptr + block_table_offset)
        kv_length = tl.load(kv_lengths_ptr + kv_length_offset)

        # Compute cache position
        cache_pos = kv_length + seq_id
        cache_offset = batch_id * k_cache_stride[0] + cache_pos * k_cache_stride[1] + head_id * k_cache_stride[2]

        # Load k and apply rotary embeddings
        k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_SIZE))
        k_rotated = k * cos - tl.rotate(k, 1) * sin

        # Store rotated k in cache
        tl.store(k_cache_ptr + cache_offset + tl.arange(0, BLOCK_SIZE), k_rotated)

# Wrapper function
def fused_rotary_embedding(q, k, cos, sin, k_cache, block_tables, kv_lengths, head_num, head_dim):
    batch_size, seq_len, _, _ = q.shape
    BLOCK_SIZE = head_dim

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    grid = (batch_size * seq_len * head_num,)
    fused_rotary_embedding_kernel_v2[grid](
        q, k, cos, sin, k_cache, block_tables, kv_lengths,
        out_q, out_k,
        q.stride(), k.stride(), cos.stride(), sin.stride(), k_cache.stride(),
        block_tables.stride(), kv_lengths.stride(),
        batch_size, seq_len, head_num, head_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out_q, out_k
