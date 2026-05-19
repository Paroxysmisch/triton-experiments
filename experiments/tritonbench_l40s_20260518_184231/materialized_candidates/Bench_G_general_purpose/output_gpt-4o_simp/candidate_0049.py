import triton
import triton.language as tl

@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    q_start_loc, q_seq_length, kv_seq_length, block_offsets,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Calculate the start index in the cache
    start_idx = q_start_loc[batch_id] + block_offsets[batch_id, block_id] * BLOCK_SIZE

    # Load data from k_states and v_states
    k_data = tl.load(k_states_ptr + start_idx)
    v_data = tl.load(v_states_ptr + start_idx)

    # Store data into k_caches and v_caches
    tl.store(k_caches_ptr + start_idx, k_data)
    tl.store(v_caches_ptr + start_idx, v_data)

@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    q_start_loc, q_seq_length, kv_seq_length, block_offsets,
    k_scales_zeros, v_scales_zeros, quant_policy: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # Calculate the start index in the cache
    start_idx = q_start_loc[batch_id] + block_offsets[batch_id, block_id] * BLOCK_SIZE

    # Load data from k_states and v_states
    k_data = tl.load(k_states_ptr + start_idx)
    v_data = tl.load(v_states_ptr + start_idx)

    # Quantize data based on the quant_policy
    if quant_policy == 4:
        # Example for int4 quantization
        k_data = (k_data - k_scales_zeros[0]) * k_scales_zeros[1]
        v_data = (v_data - v_scales_zeros[0]) * v_scales_zeros[1]
    elif quant_policy == 8:
        # Example for int8 quantization
        k_data = (k_data - k_scales_zeros[0]) * k_scales_zeros[1]
        v_data = (v_data - v_scales_zeros[0]) * v_scales_zeros[1]

    # Store data into k_caches and v_caches
    tl.store(k_caches_ptr + start_idx, k_data)
    tl.store(v_caches_ptr + start_idx, v_data)

def fill_kv_cache(
    k_states, v_states, k_caches, v_caches,
    q_start_loc, q_seq_length, kv_seq_length, block_offsets,
    k_scales_zeros=None, v_scales_zeros=None, quant_policy=0
):
    batch_size, max_num_blocks = k_states.shape[0], block_offsets.shape[1]
    BLOCK_SIZE = 128  # Define a suitable block size

    grid = (batch_size, max_num_blocks)

    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            q_start_loc, q_seq_length, kv_seq_length, block_offsets,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            q_start_loc, q_seq_length, kv_seq_length, block_offsets,
            k_scales_zeros, v_scales_zeros, quant_policy=quant_policy,
            BLOCK_SIZE=BLOCK_SIZE
        )
