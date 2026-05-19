import triton
import triton.language as tl

@triton.jit
def _quant_int4(x, scale, zero_point):
    x_scaled = (x / scale) + zero_point
    x_clamped = tl.maximum(tl.minimum(x_scaled, 7.0), -8.0)
    x_rounded = tl.round(x_clamped)
    return x_rounded.to(tl.int8)

@triton.jit
def _quant_int8(x, scale, zero_point):
    x_scaled = (x / scale) + zero_point
    x_clamped = tl.maximum(tl.minimum(x_scaled, 127.0), -128.0)
    x_rounded = tl.round(x_clamped)
    return x_rounded.to(tl.int8)

@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr,
    k_caches_ptr, v_caches_ptr,
    start_indices_ptr, seq_lengths_ptr,
    batch_size, num_heads, head_dim, max_num_blocks,
    BLOCK_SIZE_HEAD: tl.constexpr, BLOCK_SIZE_SEQ: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    head_offset = pid_block * BLOCK_SIZE_HEAD
    head_idxs = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
    seq_offset = 0
    seq_idxs = seq_offset + tl.arange(0, BLOCK_SIZE_SEQ)

    valid_head = head_idxs < num_heads
    bid = pid_batch
    nb = batch_size

    start_index = tl.load(start_indices_ptr + bid)
    seq_len = tl.load(seq_lengths_ptr + bid)

    # Grid-stride loop
    offset_block = 0
    while offset_block < max_num_blocks:
        seq_block_offset = offset_block * BLOCK_SIZE_SEQ
        seq_block_idxs = seq_block_offset + seq_idxs
        valid_seq = seq_block_idxs < seq_len
        k_src_idx = (bid * num_heads * head_dim) + (head_idxs * head_dim)
        k_src = k_states_ptr + k_src_idx + seq_block_idxs * (num_heads * head_dim * nb)
        v_src_idx = (bid * num_heads * head_dim) + (head_idxs * head_dim)
        v_src = v_states_ptr + v_src_idx + seq_block_idxs * (num_heads * head_dim * nb)

        k_dst_idx = (bid * num_heads * head_dim * max_num_blocks) + (head_idxs * head_dim * max_num_blocks)
        k_dst = k_caches_ptr + k_dst_idx + (seq_block_idxs + start_index) * head_dim
        v_dst_idx = (bid * num_heads * head_dim * max_num_blocks) + (head_idxs * head_dim * max_num_blocks)
        v_dst = v_caches_ptr + v_dst_idx + (seq_block_idxs + start_index) * head_dim

        mask = valid_head & valid_seq
        k_val = tl.load(k_src, mask=mask, other=0.0)
        v_val = tl.load(v_src, mask=mask, other=0.0)

        tl.store(k_dst, k_val, mask=mask)
        tl.store(v_dst, v_val, mask=mask)

        offset_block += 1

@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr,
    k_caches_ptr, v_caches_ptr,
    k_scale_zero_ptr, v_scale_zero_ptr,
    start_indices_ptr, seq_lengths_ptr,
    batch_size, num_heads, head_dim, max_num_blocks,
    quant_policy,
    BLOCK_SIZE_HEAD: tl.constexpr, BLOCK_SIZE_SEQ: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_block = tl.program_id(1)
    head_offset = pid_block * BLOCK_SIZE_HEAD
    head_idxs = head_offset + tl.arange(0, BLOCK_SIZE_HEAD)
    seq_offset = 0
    seq_idxs = seq_offset + tl.arange(0, BLOCK_SIZE_SEQ)

    valid_head = head_idxs < num_heads
    bid = pid_batch
    nb = batch_size

    start_index = tl.load(start_indices_ptr + bid)
    seq_len = tl.load(seq_lengths_ptr + bid)

    offset_block = 0
    while offset_block < max_num_blocks:
        seq_block_offset = offset_block * BLOCK_SIZE_SEQ
        seq_block_idxs = seq_block_offset + seq_idxs
        valid_seq = seq_block_idxs < seq_len
        k_src_idx = (bid * num_heads * head_dim) + (head_idxs * head_dim)
        k_src = k_states_ptr + k_src_idx + seq_block_idxs * (num_heads * head_dim * nb)
        v_src_idx = (bid * num_heads * head_dim) + (head_idxs * head_dim)
        v_src = v_states_ptr + v_src_idx + seq_block_idxs * (num_heads * head_dim * nb)

        k_scale = tl.load(k_scale_zero_ptr + head_idxs * 2 + 0, mask=valid_head, other=1.0)
        k_zero = tl.load(k_scale_zero_ptr + head_idxs * 2 + 1, mask=valid_head, other=0.0)
        v_scale = tl.load(v_scale_zero_ptr + head_idxs * 2 + 0, mask=valid_head, other=1.0)
        v_zero = tl.load(v_scale_zero_ptr + head_idxs * 2 + 1, mask=valid_head, other=0.0)

        k_val = tl.load(k_src, mask=valid_head & valid_seq, other=0.0)
        v_val = tl.load(v_src, mask=valid_head & valid_seq, other=0.0)

        quant_fn_k = _quant_int8 if quant_policy == 1 else _quant_int4
        quant_fn_v = _quant_int8 if quant_policy == 1 else _quant_int4

        kq = quant_fn_k(k_val, k_scale, k_zero)
        vq = quant_fn_v(v_val, v_scale, v_zero)

        k_dst_idx = (bid * num_heads * head_dim * max_num_blocks) + (head_idxs * head_dim * max_num_blocks)
        k_dst = k_caches_ptr + k_dst_idx + (seq_block_idxs + start_index) * head_dim
        v_dst_idx = (bid * num_heads * head_dim * max_num_blocks) + (head_idxs * head_dim * max_num_blocks)
        v_dst = v_caches_ptr + v_dst_idx + (seq_block_idxs + start_index) * head_dim

        tl.store(k_dst, kq, mask=valid_head & valid_seq)
        tl.store(v_dst, vq, mask=valid_head & valid_seq)

        offset_block += 1

def fill_kv_cache(
    k_states, v_states,
    k_caches, v_caches,
    start_indices, seq_lengths,
    batch_size, num_heads, head_dim, max_num_blocks,
    quant_policy=0,
    k_scales_zeros=None, v_scales_zeros=None
):
    grid = (batch_size, (num_heads + 31) // 32)  # example
    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states,
            k_caches, v_caches,
            start_indices, seq_lengths,
            batch_size, num_heads, head_dim, max_num_blocks,
            BLOCK_SIZE_HEAD=32, BLOCK_SIZE_SEQ=32
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states,
            k_caches, v_caches,
            k_scales_zeros, v_scales_zeros,
            start_indices, seq_lengths,
            batch_size, num_heads, head_dim, max_num_blocks,
            quant_policy,
            BLOCK_SIZE_HEAD=32, BLOCK_SIZE_SEQ=32
        )
