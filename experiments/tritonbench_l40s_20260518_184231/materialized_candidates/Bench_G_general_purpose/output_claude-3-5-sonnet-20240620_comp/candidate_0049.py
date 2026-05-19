import triton
import triton.language as tl
import torch

@triton.jit
def _quant_int8(
    x: tl.tensor,
    scale: tl.tensor,
    zero_point: tl.tensor,
):
    # Scale and quantize to int8
    x_scaled = x / scale
    x_rounded = tl.round(x_scaled + zero_point)
    return tl.clip(x_rounded, -128.0, 127.0)

@triton.jit
def _quant_int4(
    x: tl.tensor,
    scale: tl.tensor,
    zero_point: tl.tensor,
):
    # Scale and quantize to int4
    x_scaled = x / scale
    x_rounded = tl.round(x_scaled + zero_point)
    return tl.clip(x_rounded, -8.0, 7.0)

@triton.jit
def _fill_kv_cache_kernel(
    k_states_ptr, v_states_ptr,
    k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    batch_size, max_num_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    batch_id = pid // max_num_blocks
    block_id = pid % max_num_blocks

    if batch_id >= batch_size:
        return

    # Load metadata
    query_start = tl.load(query_start_loc_ptr + batch_id)
    seq_len = tl.load(seq_lens_ptr + batch_id)

    # Check if this block needs processing
    if block_id * BLOCK_SIZE >= seq_len:
        return

    # Calculate offsets
    block_start = block_id * BLOCK_SIZE
    block_size = tl.minimum(BLOCK_SIZE, seq_len - block_start)
    
    # Create block-level offsets
    offs_block = tl.arange(0, BLOCK_SIZE)
    mask = offs_block < block_size

    # Load and store for each head dimension
    for head_idx in range(0, HEAD_DIM, 32):
        offs_head = tl.arange(0, tl.minimum(32, HEAD_DIM - head_idx))
        
        # Load states
        k_state = tl.load(k_states_ptr + batch_id * seq_len * HEAD_DIM + 
                         (block_start + offs_block[:, None]) * HEAD_DIM + 
                         (head_idx + offs_head[None, :]),
                         mask=mask[:, None], other=0.0)
        v_state = tl.load(v_states_ptr + batch_id * seq_len * HEAD_DIM + 
                         (block_start + offs_block[:, None]) * HEAD_DIM + 
                         (head_idx + offs_head[None, :]),
                         mask=mask[:, None], other=0.0)

        # Store to cache
        tl.store(k_cache_ptr + batch_id * max_num_blocks * BLOCK_SIZE * HEAD_DIM + 
                block_id * BLOCK_SIZE * HEAD_DIM +
                offs_block[:, None] * HEAD_DIM + 
                (head_idx + offs_head[None, :]),
                k_state, mask=mask[:, None])
        tl.store(v_cache_ptr + batch_id * max_num_blocks * BLOCK_SIZE * HEAD_DIM + 
                block_id * BLOCK_SIZE * HEAD_DIM +
                offs_block[:, None] * HEAD_DIM + 
                (head_idx + offs_head[None, :]),
                v_state, mask=mask[:, None])

@triton.jit
def _fill_kv_cache_quant_kernel(
    k_states_ptr, v_states_ptr,
    k_cache_ptr, v_cache_ptr,
    k_scales_zeros_ptr, v_scales_zeros_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    batch_size, max_num_blocks,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,  # 4 for int4, 8 for int8
):
    # Similar structure to non-quant kernel
    pid = tl.program_id(0)
    batch_id = pid // max_num_blocks
    block_id = pid % max_num_blocks

    if batch_id >= batch_size:
        return

    query_start = tl.load(query_start_loc_ptr + batch_id)
    seq_len = tl.load(seq_lens_ptr + batch_id)

    if block_id * BLOCK_SIZE >= seq_len:
        return

    block_start = block_id * BLOCK_SIZE
    block_size = tl.minimum(BLOCK_SIZE, seq_len - block_start)
    
    offs_block = tl.arange(0, BLOCK_SIZE)
    mask = offs_block < block_size

    for head_idx in range(0, HEAD_DIM, 32):
        offs_head = tl.arange(0, tl.minimum(32, HEAD_DIM - head_idx))
        
        # Load states
        k_state = tl.load(k_states_ptr + batch_id * seq_len * HEAD_DIM + 
                         (block_start + offs_block[:, None]) * HEAD_DIM + 
                         (head_idx + offs_head[None, :]),
                         mask=mask[:, None], other=0.0)
        v_state = tl.load(v_states_ptr + batch_id * seq_len * HEAD_DIM + 
                         (block_start + offs_block[:, None]) * HEAD_DIM + 
                         (head_idx + offs_head[None, :]),
                         mask=mask[:, None], other=0.0)

        # Load scales and zero points
        k_scale = tl.load(k_scales_zeros_ptr + batch_id * 2 * HEAD_DIM + head_idx)
        k_zero = tl.load(k_scales_zeros_ptr + batch_id * 2 * HEAD_DIM + HEAD_DIM + head_idx)
        v_scale = tl.load(v_scales_zeros_ptr + batch_id * 2 * HEAD_DIM + head_idx)
        v_zero = tl.load(v_scales_zeros_ptr + batch_id * 2 * HEAD_DIM + HEAD_DIM + head_idx)

        # Quantize
        if QUANT_TYPE == 4:
            k_quant = _quant_int4(k_state, k_scale, k_zero)
            v_quant = _quant_int4(v_state, v_scale, v_zero)
        else:  # int8
            k_quant = _quant_int8(k_state, k_scale, k_zero)
            v_quant = _quant_int8(v_state, v_scale, v_zero)

        # Store quantized values
        tl.store(k_cache_ptr + batch_id * max_num_blocks * BLOCK_SIZE * HEAD_DIM + 
                block_id * BLOCK_SIZE * HEAD_DIM +
                offs_block[:, None] * HEAD_DIM + 
                (head_idx + offs_head[None, :]),
                k_quant, mask=mask[:, None])
        tl.store(v_cache_ptr + batch_id * max_num_blocks * BLOCK_SIZE * HEAD_DIM + 
                block_id * BLOCK_SIZE * HEAD_DIM +
                offs_block[:, None] * HEAD_DIM + 
                (head_idx + offs_head[None, :]),
                v_quant, mask=mask[:, None])

def fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scales_zeros: torch.Tensor = None,
    v_scales_zeros: torch.Tensor = None,
    quant_policy: int = 0,  # 0: no quant, 4: int4, 8: int8
):
    batch_size = k_states.shape[0]
    head_dim = k_states.shape[-1]
    max_num_blocks = k_cache.shape[1]
    block_size = k_cache.shape[2]

    grid = (batch_size * max_num_blocks,)

    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states,
            k_cache, v_cache,
            query_start_loc,
            seq_lens,
            batch_size, max_num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states,
            k_cache, v_cache,
            k_scales_zeros, v_scales_zeros,
            query_start_loc,
            seq_lens,
            batch_size, max_num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            QUANT_TYPE=quant_policy,
        )
