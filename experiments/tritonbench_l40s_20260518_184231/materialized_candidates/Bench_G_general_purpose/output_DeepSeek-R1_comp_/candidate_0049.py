import torch
import triton
import triton.language as tl

@triton.jit
def _fill_kv_cache_kernel(
    # Pointers to tensors
    k_states_ptr,
    v_states_ptr,
    k_cache_ptr,
    v_cache_ptr,
    start_loc_ptr,
    seq_lens_ptr,
    block_size,
    # Strides for k/v_states
    k_states_batch_stride,
    k_states_seq_stride,
    k_states_head_stride,
    k_states_dim_stride,
    # Strides for k_cache
    k_cache_block_stride,
    k_cache_batch_stride,
    k_cache_head_stride,
    k_cache_block_off_stride,
    k_cache_dim_stride,
    # Strides for v_cache
    v_cache_block_stride,
    v_cache_batch_stride,
    v_cache_head_stride,
    v_cache_block_off_stride,
    v_cache_dim_stride,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    start_block = tl.load(start_loc_ptr + batch_idx)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if block_idx >= num_blocks:
        return
    
    cache_block = start_block + block_idx
    start_token = block_idx * BLOCK_SIZE
    end_token = min(start_token + BLOCK_SIZE, seq_len)
    valid_tokens = end_token - start_token

    # Calculate base pointers for this block
    k_cache_base = (k_cache_ptr +
                    cache_block * k_cache_block_stride +
                    batch_idx * k_cache_batch_stride +
                    head_idx * k_cache_head_stride)
    
    v_cache_base = (v_cache_ptr +
                    cache_block * v_cache_block_stride +
                    batch_idx * v_cache_batch_stride +
                    head_idx * v_cache_head_stride)

    # Process each valid token in the block
    for s in range(valid_tokens):
        token_idx = start_token + s
        
        # Calculate state pointers
        k_state_ptr = (k_states_ptr +
                       batch_idx * k_states_batch_stride +
                       token_idx * k_states_seq_stride +
                       head_idx * k_states_head_stride)
        
        v_state_ptr = (v_states_ptr +
                       batch_idx * k_states_batch_stride +
                       token_idx * k_states_seq_stride +
                       head_idx * k_states_head_stride)

        # Vectorized copy of head dimension
        dim_offs = tl.arange(0, head_dim)
        dim_mask = dim_offs < head_dim
        
        # Load state values
        k_vals = tl.load(k_state_ptr + dim_offs * k_states_dim_stride, mask=dim_mask)
        v_vals = tl.load(v_state_ptr + dim_offs * k_states_dim_stride, mask=dim_mask)

        # Calculate cache positions
        k_cache_pos = k_cache_base + s * k_cache_block_off_stride
        v_cache_pos = v_cache_base + s * v_cache_block_off_stride

        # Store values in cache
        tl.store(k_cache_pos + dim_offs * k_cache_dim_stride, k_vals, mask=dim_mask)
        tl.store(v_cache_pos + dim_offs * v_cache_dim_stride, v_vals, mask=dim_mask)

@triton.jit
def _quant_int8(x, scale, zero):
    x = tl.math.llrint(x / scale + zero)
    return tl.where(x < -128, -128, tl.where(x > 127, 127, x)).to(tl.int8)

@triton.jit
def _quant_int4(x, scale, zero):
    x = tl.math.llrint(x / scale + zero)
    x = tl.where(x < -8, -8, tl.where(x > 7, 7, x))
    return ((x[::2] & 0xF) | (x[1::2] & 0xF) << 4).to(tl.int8)

@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr,
    v_states_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scales_ptr,
    v_scales_ptr,
    k_zeros_ptr,
    v_zeros_ptr,
    start_loc_ptr,
    seq_lens_ptr,
    block_size,
    quant_policy,
    # Strides parameters same as non-quant kernel
    ...  # Same stride parameters as previous kernel
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    start_block = tl.load(start_loc_ptr + batch_idx)
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    if block_idx >= num_blocks:
        return

    cache_block = start_block + block_idx
    start_token = block_idx * BLOCK_SIZE
    end_token = min(start_token + BLOCK_SIZE, seq_len)
    valid_tokens = end_token - start_token

    # Calculate quantization parameters
    max_val = tl.zeros((1,), tl.float32)
    min_val = tl.zeros((1,), tl.float32)
    for s in range(valid_tokens):
        ...  # Similar to non-quant kernel for loading values
        max_val = tl.maximum(max_val, tl.max(k_vals))
        min_val = tl.minimum(min_val, tl.min(k_vals))

    scale = (max_val - min_val) / (255 if quant_policy & 1 else 15)
    zero = tl.cast((min_val / scale).to(tl.float32), tl.int8)

    # Quantize and store
    for s in range(valid_tokens):
        ...  # Load values as in non-quant kernel
        if quant_policy & 1:
            q = _quant_int8(k_vals, scale, zero)
        else:
            q = _quant_int4(k_vals, scale, zero)
        ...  # Store quantized values

    # Store scale and zero
    tl.store(k_scales_ptr + cache_block, scale)
    tl.store(k_zeros_ptr + cache_block, zero)
    ...  # Repeat for v_states

def fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scales: torch.Tensor = None,
    v_scales: torch.Tensor = None,
    k_zeros: torch.Tensor = None,
    v_zeros: torch.Tensor = None,
    quant_policy: int = 0,
    block_size: int = 128,
):
    batch_size, num_heads, head_dim = k_states.shape
    max_num_blocks = ((seq_lens.max().item() + block_size - 1) // block_size)

    grid = (batch_size, max_num_blocks, num_heads)
    
    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states, k_cache, v_cache,
            start_loc, seq_lens, block_size,
            *k_states.stride(),
            *k_cache.stride(),
            *v_cache.stride(),
            head_dim,
            BLOCK_SIZE=block_size
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states, k_cache, v_cache,
            k_scales, v_scales, k_zeros, v_zeros,
            start_loc, seq_lens, block_size, quant_policy,
            *k_states.stride(),
            *k_cache.stride(),
            *v_cache.stride(),
            head_dim,
            BLOCK_SIZE=block_size
        )
