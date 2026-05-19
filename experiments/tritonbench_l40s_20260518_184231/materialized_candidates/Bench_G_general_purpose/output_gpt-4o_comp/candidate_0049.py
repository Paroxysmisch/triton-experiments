import triton
import triton.language as tl

# Helper functions for quantization
@triton.jit
def _quant_int4(x, scale, zero_point):
    # Quantize to int4 using scale and zero-point
    return tl.cast(tl.clamp((x / scale + zero_point), 0, 15), tl.int8)

@triton.jit
def _quant_int8(x, scale, zero_point):
    # Quantize to int8 using scale and zero-point
    return tl.cast(tl.clamp((x / scale + zero_point), -128, 127), tl.int8)

# Kernel for direct copy (non-quantized)
@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    batch_size, num_heads, seq_len, head_dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim
    v_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim

    k_states = tl.load(k_states_ptr + k_offsets, mask=offsets < head_dim)
    v_states = tl.load(v_states_ptr + v_offsets, mask=offsets < head_dim)

    k_cache_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim
    v_cache_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim

    tl.store(k_caches_ptr + k_cache_offsets, k_states, mask=offsets < head_dim)
    tl.store(v_caches_ptr + v_cache_offsets, v_states, mask=offsets < head_dim)

# Kernel for quantized copy
@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr, k_caches_ptr, v_caches_ptr,
    k_scales_zeros_ptr, v_scales_zeros_ptr,
    batch_size, num_heads, seq_len, head_dim, quant_policy, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim
    v_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim

    k_states = tl.load(k_states_ptr + k_offsets, mask=offsets < head_dim)
    v_states = tl.load(v_states_ptr + v_offsets, mask=offsets < head_dim)

    # Load scales and zeros
    k_scale, k_zero = tl.load(k_scales_zeros_ptr + head_id * 2)
    v_scale, v_zero = tl.load(v_scales_zeros_ptr + head_id * 2)

    # Quantize
    if quant_policy == 1:  # int4
        k_quant = _quant_int4(k_states, k_scale, k_zero)
        v_quant = _quant_int4(v_states, v_scale, v_zero)
    else:  # int8
        k_quant = _quant_int8(k_states, k_scale, k_zero)
        v_quant = _quant_int8(v_states, v_scale, v_zero)

    k_cache_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim
    v_cache_offsets = offsets + head_id * head_dim + batch_id * seq_len * head_dim

    tl.store(k_caches_ptr + k_cache_offsets, k_quant, mask=offsets < head_dim)
    tl.store(v_caches_ptr + v_cache_offsets, v_quant, mask=offsets < head_dim)

# Main function
def fill_kv_cache(
    k_states, v_states, k_caches, v_caches,
    batch_size, num_heads, seq_len, head_dim,
    k_scales_zeros=None, v_scales_zeros=None, quant_policy=0
):
    BLOCK_SIZE = 128  # Example block size, adjust based on your needs
    grid = (batch_size * num_heads, )

    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            batch_size, num_heads, seq_len, head_dim, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states, k_caches, v_caches,
            k_scales_zeros, v_scales_zeros,
            batch_size, num_heads, seq_len, head_dim, quant_policy, BLOCK_SIZE=BLOCK_SIZE
        )
