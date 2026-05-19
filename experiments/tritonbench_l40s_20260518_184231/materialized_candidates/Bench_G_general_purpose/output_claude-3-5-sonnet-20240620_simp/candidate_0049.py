import torch
import triton
import triton.language as tl

# Kernel for non-quantized version
@triton.jit
def _fill_kv_cache_kernel(
    # Pointers to tensors
    k_states_ptr, v_states_ptr,      # [*,H,S,D]
    k_cache_ptr, v_cache_ptr,        # [B,H,L,D]
    q_start_loc_ptr,                 # [B]
    block_offsets_ptr,               # [B,L]
    # Dimensions and strides
    batch_size, num_heads, seq_length, head_dim,
    k_states_batch_stride, k_states_head_stride, k_states_seq_stride, k_states_head_dim_stride,
    v_states_batch_stride, v_states_head_stride, v_states_seq_stride, v_states_head_dim_stride,
    k_cache_batch_stride, k_cache_head_stride, k_cache_seq_stride, k_cache_head_dim_stride,
    v_cache_batch_stride, v_cache_head_stride, v_cache_seq_stride, v_cache_head_dim_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)  # Batch index
    head_id = tl.program_id(1)  # Head index

    # Compute start location for this sequence
    q_start = tl.load(q_start_loc_ptr + pid)
    
    # Offsets for loading/storing
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < head_dim
    
    for seq_idx in range(0, seq_length):
        # Get block offset
        block_offset = tl.load(block_offsets_ptr + pid * seq_length + seq_idx)
        
        # Load from states
        k_state_ptr = k_states_ptr + (
            pid * k_states_batch_stride +
            head_id * k_states_head_stride +
            seq_idx * k_states_seq_stride
        )
        v_state_ptr = v_states_ptr + (
            pid * v_states_batch_stride +
            head_id * v_states_head_stride +
            seq_idx * v_states_seq_stride
        )
        
        # Load values
        k_vals = tl.load(k_state_ptr + offs_d * k_states_head_dim_stride, mask=mask_d)
        v_vals = tl.load(v_state_ptr + offs_d * v_states_head_dim_stride, mask=mask_d)
        
        # Store to cache
        cache_idx = q_start + seq_idx
        k_cache_ptr_offset = (
            pid * k_cache_batch_stride +
            head_id * k_cache_head_stride +
            cache_idx * k_cache_seq_stride
        )
        v_cache_ptr_offset = (
            pid * v_cache_batch_stride +
            head_id * v_cache_head_stride +
            cache_idx * v_cache_seq_stride
        )
        
        # Store values
        tl.store(k_cache_ptr + k_cache_ptr_offset + offs_d * k_cache_head_dim_stride, k_vals, mask=mask_d)
        tl.store(v_cache_ptr + v_cache_ptr_offset + offs_d * v_cache_head_dim_stride, v_vals, mask=mask_d)

# Kernel for quantized version
@triton.jit
def _fill_kv_cache_quant_kernel(
    # Pointers to tensors
    k_states_ptr, v_states_ptr,      # [*,H,S,D]
    k_cache_ptr, v_cache_ptr,        # [B,H,L,D]
    k_scales_zeros_ptr, v_scales_zeros_ptr,  # Scale and zero points for quantization
    q_start_loc_ptr,                 # [B]
    block_offsets_ptr,               # [B,L]
    # Dimensions and strides
    batch_size, num_heads, seq_length, head_dim,
    k_states_batch_stride, k_states_head_stride, k_states_seq_stride, k_states_head_dim_stride,
    v_states_batch_stride, v_states_head_stride, v_states_seq_stride, v_states_head_dim_stride,
    k_cache_batch_stride, k_cache_head_stride, k_cache_seq_stride, k_cache_head_dim_stride,
    v_cache_batch_stride, v_cache_head_stride, v_cache_seq_stride, v_cache_head_dim_stride,
    quant_policy: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    head_id = tl.program_id(1)
    
    q_start = tl.load(q_start_loc_ptr + pid)
    offs_d = tl.arange(0, BLOCK_SIZE)
    mask_d = offs_d < head_dim
    
    for seq_idx in range(0, seq_length):
        block_offset = tl.load(block_offsets_ptr + pid * seq_length + seq_idx)
        
        # Load values
        k_state_ptr = k_states_ptr + (
            pid * k_states_batch_stride +
            head_id * k_states_head_stride +
            seq_idx * k_states_seq_stride
        )
        v_state_ptr = v_states_ptr + (
            pid * v_states_batch_stride +
            head_id * v_states_head_stride +
            seq_idx * v_states_seq_stride
        )
        
        k_vals = tl.load(k_state_ptr + offs_d * k_states_head_dim_stride, mask=mask_d)
        v_vals = tl.load(v_state_ptr + offs_d * v_states_head_dim_stride, mask=mask_d)
        
        # Load scales and zero points
        k_scale = tl.load(k_scales_zeros_ptr + pid * num_heads * 2 + head_id * 2)
        k_zero = tl.load(k_scales_zeros_ptr + pid * num_heads * 2 + head_id * 2 + 1)
        v_scale = tl.load(v_scales_zeros_ptr + pid * num_heads * 2 + head_id * 2)
        v_zero = tl.load(v_scales_zeros_ptr + pid * num_heads * 2 + head_id * 2 + 1)
        
        # Quantize
        if quant_policy == 4:
            k_vals = tl.math.round((k_vals / k_scale) + k_zero).to(tl.int4)
            v_vals = tl.math.round((v_vals / v_scale) + v_zero).to(tl.int4)
        else:  # int8
            k_vals = tl.math.round((k_vals / k_scale) + k_zero).to(tl.int8)
            v_vals = tl.math.round((v_vals / v_scale) + v_zero).to(tl.int8)
        
        # Store to cache
        cache_idx = q_start + seq_idx
        k_cache_ptr_offset = (
            pid * k_cache_batch_stride +
            head_id * k_cache_head_stride +
            cache_idx * k_cache_seq_stride
        )
        v_cache_ptr_offset = (
            pid * v_cache_batch_stride +
            head_id * v_cache_head_stride +
            cache_idx * v_cache_seq_stride
        )
        
        tl.store(k_cache_ptr + k_cache_ptr_offset + offs_d * k_cache_head_dim_stride, k_vals, mask=mask_d)
        tl.store(v_cache_ptr + v_cache_ptr_offset + offs_d * v_cache_head_dim_stride, v_vals, mask=mask_d)

def fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_caches: torch.Tensor,
    v_caches: torch.Tensor,
    q_start_loc: torch.Tensor,
    q_seq_length: int,
    kv_seq_length: int,
    block_offsets: torch.Tensor,
    k_scales_zeros: torch.Tensor = None,
    v_scales_zeros: torch.Tensor = None,
    quant_policy: int = 0
):
    batch_size = k_caches.shape[0]
    num_heads = k_caches.shape[1]
    head_dim = k_caches.shape[-1]
    
    # Get strides for efficient memory access
    k_states_strides = k_states.stride()
    v_states_strides = v_states.stride()
    k_cache_strides = k_caches.stride()
    v_cache_strides = v_caches.stride()
    
    # Configure grid and block sizes
    grid = (batch_size, num_heads)
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    # Launch appropriate kernel based on quantization policy
    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states, v_states,
            k_caches, v_caches,
            q_start_loc, block_offsets,
            batch_size, num_heads, kv_seq_length, head_dim,
            *k_states_strides, *v_states_strides,
            *k_cache_strides, *v_cache_strides,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states, v_states,
            k_caches, v_caches,
            k_scales_zeros, v_scales_zeros,
            q_start_loc, block_offsets,
            batch_size, num_heads, kv_seq_length, head_dim,
            *k_states_strides, *v_states_strides,
            *k_cache_strides, *v_cache_strides,
            quant_policy=quant_policy,
            BLOCK_SIZE=BLOCK_SIZE
        )
