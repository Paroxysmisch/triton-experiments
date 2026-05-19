import triton
import triton.language as tl

# Triton kernel for filling key and value states into cache (non-quantized)
@triton.jit
def _fill_kv_cache_kernel(
    k_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    k_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    query_start: tl.tensor,  # (batch_size,)
    seq_lengths: tl.tensor,  # (batch_size,)
    k_scales_zeros: tl.tensor = None,  # (batch_size, num_heads)
    v_scales_zeros: tl.tensor = None,  # (batch_size, num_heads)
    max_num_blocks: int,
    block_size: int,
    head_dim: int,
    num_heads: int,
    max_seq_len: int,
    batch_size: int,
    quant_policy: int = 0,
):
    # Compute indices
    b = tl.program_id(0)  # batch index
    seq_len = seq_lengths[b]
    head_idx = tl.program_id(1)  # head index
    kv_idx = tl.program_id(2)  # kv index (0 for keys, 1 for values)

    # Compute start and end indices for the block
    start_idx = b * max_seq_len + query_start[b]
    end_idx = start_idx + seq_len

    # Compute block indices
    block_idx = tl.program_id(3)
    block_start = block_idx * block_size
    block_end = min(block_start + block_size, seq_len)

    # Compute indices within the block
    block_offset = tl.arange(0, block_size)
    indices = block_start + block_offset

    # Compute the destination index in the cache
    cache_offset = tl.arange(0, block_size)
    cache_indices = start_idx + cache_offset

    # Ensure valid indices
    valid_indices = indices < seq_len

    # Load data from states
    k_state = tl.load(k_states + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + indices * head_dim)
    v_state = tl.load(v_states + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + indices * head_dim)

    # Store data in cache
    tl.store(k_caches + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + cache_indices * head_dim, k_state, mask=valid_indices)
    tl.store(v_caches + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + cache_indices * head_dim, v_state, mask=valid_indices)

# Triton kernel for filling key and value states into cache (quantized)
@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    k_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    query_start: tl.tensor,  # (batch_size,)
    seq_lengths: tl.tensor,  # (batch_size,)
    k_scales_zeros: tl.tensor,  # (batch_size, num_heads)
    v_scales_zeros: tl.tensor,  # (batch_size, num_heads)
    max_num_blocks: int,
    block_size: int,
    head_dim: int,
    num_heads: int,
    max_seq_len: int,
    batch_size: int,
    quant_policy: int = 0,
):
    # Compute indices
    b = tl.program_id(0)  # batch index
    seq_len = seq_lengths[b]
    head_idx = tl.program_id(1)  # head index
    kv_idx = tl.program_id(2)  # kv index (0 for keys, 1 for values)

    # Compute start and end indices for the block
    start_idx = b * max_seq_len + query_start[b]
    end_idx = start_idx + seq_len

    # Compute block indices
    block_idx = tl.program_id(3)
    block_start = block_idx * block_size
    block_end = min(block_start + block_size, seq_len)

    # Compute indices within the block
    block_offset = tl.arange(0, block_size)
    indices = block_start + block_offset

    # Compute the destination index in the cache
    cache_offset = tl.arange(0, block_size)
    cache_indices = start_idx + cache_offset

    # Ensure valid indices
    valid_indices = indices < seq_len

    # Load data from states
    k_state = tl.load(k_states + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + indices * head_dim)
    v_state = tl.load(v_states + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + indices * head_dim)

    # Quantize data
    k_scale = tl.load(k_scales_zeros + b * num_heads + head_idx, mask=valid_indices)
    v_scale = tl.load(v_scales_zeros + b * num_heads + head_idx, mask=valid_indices)
    k_zero_point = tl.zeros_like(k_scale)
    v_zero_point = tl.zeros_like(v_scale)

    k_quant = _quant_int4(k_state, k_scale, k_zero_point)
    v_quant = _quant_int4(v_state, v_scale, v_zero_point)

    # Store data in cache
    tl.store(k_caches + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + cache_indices * head_dim, k_quant, mask=valid_indices)
    tl.store(v_caches + b * max_seq_len * num_heads * head_dim + head_idx * head_dim + cache_indices * head_dim, v_quant, mask=valid_indices)

# Triton kernel for quantizing int4 data
@triton.jit
def _quant_int4(x: tl.tensor, scale: tl.tensor, zero_point: tl.tensor) -> tl.tensor:
    # Quantize data to int4
    quantized = tl.bitcast(x / scale + zero_point, tl.int8)
    quantized = tl.bitcast(quantized, tl.int4)
    return quantized

# Triton kernel for quantizing int8 data
@triton.jit
def _quant_int8(x: tl.tensor, scale: tl.tensor, zero_point: tl.tensor) -> tl.tensor:
    # Quantize data to int8
    quantized = tl.bitcast(x / scale + zero_point, tl.int8)
    return quantized

# Triton function to fill key and value states into cache
@triton.jit
def fill_kv_cache(
    k_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_states: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    k_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    v_caches: tl.tensor,  # (batch_size, max_seq_len, num_heads, head_dim)
    query_start: tl.tensor,  # (batch_size,)
    seq_lengths: tl.tensor,  # (batch_size,)
    k_scales_zeros: tl.tensor = None,  # (batch_size, num_heads)
    v_scales_zeros: tl.tensor = None,  # (batch_size, num_heads)
    max_num_blocks: int,
    block_size: int,
    head_dim: int,
    num_heads: int,
    max_seq_len: int,
    batch_size: int,
    quant_policy: int = 0,
):
    # Determine the kernel to use based on quant_policy
    if quant_policy == 0:
        _fill_kv_cache_kernel[kernel_config](k_states, v_states, k_caches, v_caches, query_start, seq_lengths, k_scales_zeros, v_scales_zeros, max_num_blocks, block_size, head_dim, num_heads, max_seq_len, batch_size, quant_policy)
    else:
        _fill_kv_cache_quant_kernel[kernel_config](k_states, v_states, k_caches, v_caches, query_start, seq_lengths, k_scales_zeros, v_scales_zeros, max_num_blocks, block_size, head_dim, num_heads, max_seq_len, batch_size, quant_policy)
