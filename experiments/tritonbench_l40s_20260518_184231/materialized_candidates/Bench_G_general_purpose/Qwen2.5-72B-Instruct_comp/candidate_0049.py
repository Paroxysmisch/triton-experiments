import triton
import triton.language as tl

# Helper functions for quantization
@triton.jit
def _quant_int4(x, scale, zero_point):
    return tl.cast(tl.round((x / scale) + zero_point), tl.int4)

@triton.jit
def _quant_int8(x, scale, zero_point):
    return tl.cast(tl.round((x / scale) + zero_point), tl.int8)

# Kernel for filling cache without quantization
@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    query_start_ptr, seq_len_ptr, batch_size, max_num_blocks, head_dim, block_size,
    k_scales_zeros_ptr=None, v_scales_zeros_ptr=None, quant_policy=0
):
    pid = tl.program_id(axis=0)
    bid = pid // max_num_blocks
    block_id = pid % max_num_blocks

    if bid >= batch_size:
        return

    query_start = tl.load(query_start_ptr + bid)
    seq_len = tl.load(seq_len_ptr + bid)
    end = query_start + seq_len

    if block_id * block_size >= end:
        return

    start = max(query_start, block_id * block_size)
    end = min(end, (block_id + 1) * block_size)

    for i in range(start, end):
        k_idx = bid * head_dim + i * head_dim
        v_idx = bid * head_dim + i * head_dim
        k_cache_idx = bid * max_num_blocks * head_dim + block_id * head_dim + (i - start) * head_dim
        v_cache_idx = bid * max_num_blocks * head_dim + block_id * head_dim + (i - start) * head_dim

        k_state = tl.load(k_states_ptr + k_idx, mask=i < end, other=0.0)
        v_state = tl.load(v_states_ptr + v_idx, mask=i < end, other=0.0)

        if quant_policy == 0:
            tl.store(k_caches_ptr + k_cache_idx, k_state, mask=i < end)
            tl.store(v_caches_ptr + v_cache_idx, v_state, mask=i < end)
        else:
            k_scale, k_zero_point = tl.load(k_scales_zeros_ptr + bid * 2 + 0), tl.load(k_scales_zeros_ptr + bid * 2 + 1)
            v_scale, v_zero_point = tl.load(v_scales_zeros_ptr + bid * 2 + 0), tl.load(v_scales_zeros_ptr + bid * 2 + 1)

            k_quant = _quant_int4(k_state, k_scale, k_zero_point) if quant_policy == 1 else _quant_int8(k_state, k_scale, k_zero_point)
            v_quant = _quant_int4(v_state, v_scale, v_zero_point) if quant_policy == 1 else _quant_int8(v_state, v_scale, v_zero_point)

            tl.store(k_caches_ptr + k_cache_idx, k_quant, mask=i < end)
            tl.store(v_caches_ptr + v_cache_idx, v_quant, mask=i < end)

# Kernel for filling cache with quantization
@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    query_start_ptr, seq_len_ptr, batch_size, max_num_blocks, head_dim, block_size,
    k_scales_zeros_ptr, v_scales_zeros_ptr, quant_policy
):
    _fill_kv_cache_kernel(
        k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
        query_start_ptr, seq_len_ptr, batch_size, max_num_blocks, head_dim, block_size,
        k_scales_zeros_ptr, v_scales_zeros_ptr, quant_policy
    )

# Wrapper function to call the appropriate kernel
def fill_kv_cache(
    k_states, v_states, k_caches, v_caches,
    query_start, seq_len, batch_size, max_num_blocks, head_dim, block_size,
    k_scales_zeros=None, v_scales_zeros=None, quant_policy=0
):
    grid = (batch_size * max_num_blocks,)

    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            query_start, seq_len, batch_size, max_num_blocks, head_dim, block_size,
            k_scales_zeros, v_scales_zeros, quant_policy
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            query_start, seq_len, batch_size, max_num_blocks, head_dim, block_size,
            k_scales_zeros, v_scales_zeros, quant_policy
        )
