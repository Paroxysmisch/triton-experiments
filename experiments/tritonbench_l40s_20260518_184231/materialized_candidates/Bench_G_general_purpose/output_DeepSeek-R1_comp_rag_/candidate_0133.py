import triton
import triton.language as tl
import torch
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64, 'BLOCK_V': 64, 'BLOCK_T': 32}, num_warps=1),
        triton.Config({'BLOCK_K': 128, 'BLOCK_V': 128, 'BLOCK_T': 32}, num_warps=2),
        triton.Config({'BLOCK_K': 256, 'BLOCK_V': 256, 'BLOCK_T': 32}, num_warps=4),
        triton.Config({'BLOCK_K': 512, 'BLOCK_V': 512, 'BLOCK_T': 32}, num_warps=8),
        triton.Config({'BLOCK_K': 1024, 'BLOCK_V': 1024, 'BLOCK_T': 32}, num_warps=16),
        triton.Config({'BLOCK_K': 2048, 'BLOCK_V': 2048, 'BLOCK_T': 32}, num_warps=32),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    # Pointers to tensors
    k_ptr, v_ptr, d_ptr, h_ptr, v_new_ptr,
    initial_state_ptr,
    final_state_ptr,
    # Tensor dimensions
    BT, BK, BV,
    # Strides for k, v, d
    k_batch_stride, k_k_stride, k_v_stride,
    v_batch_stride, v_k_stride, v_v_stride,
    d_batch_stride, d_k_stride, d_v_stride,
    h_batch_stride, h_k_stride, h_v_stride,
    v_new_batch_stride, v_new_k_stride, v_new_v_stride,
    # Parameters
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # Grid indices
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    # Create block pointers for initial and final states if needed
    if USE_INITIAL_STATE:
        h_start = initial_state_ptr + i_bh * h_batch_stride + i_k * h_k_stride + i_v * h_v_stride
        h_block_ptr = tl.make_block_ptr(
            base=h_start,
            shape=(BLOCK_K, BLOCK_V),
            strides=(h_k_stride, h_v_stride),
            offsets=(0, 0),
            block_shape=(BLOCK_K, BLOCK_V),
            order=(1, 0)
        )
        b_h = tl.load(h_block_ptr, boundary_check=(0, 1))
    else:
        b_h = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    
    # Block pointers for k, v, d
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + i_bh * k_batch_stride,
        shape=(BT, BK, BV),
        strides=(k_k_stride, k_v_stride, 1),
        offsets=(0, i_k * BLOCK_K, i_v * BLOCK_V),
        block_shape=(BLOCK_T, BLOCK_K, BLOCK_V),
        order=(2, 1, 0)
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + i_bh * v_batch_stride,
        shape=(BT, BK, BV),
        strides=(v_k_stride, v_v_stride, 1),
        offsets=(0, i_k * BLOCK_K, i_v * BLOCK_V),
        block_shape=(BLOCK_T, BLOCK_K, BLOCK_V),
        order=(2, 1, 0)
    )
    
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + i_bh * d_batch_stride,
        shape=(BT, BK, BV),
        strides=(d_k_stride, d_v_stride, 1),
        offsets=(0, i_k * BLOCK_K, i_v * BLOCK_V),
        block_shape=(BLOCK_T, BLOCK_K, BLOCK_V),
        order=(2, 1, 0)
    )
    
    # Initialize cumulative sum
    b_h_cumsum = b_h
    NT = tl.cdiv(BT, BLOCK_T)
    
    for t in range(NT):
        # Load current k, v, d blocks
        b_k = tl.load(k_block_ptr)
        b_v = tl.load(v_block_ptr)
        b_d = tl.load(d_block_ptr)
        
        # Compute intermediate values
        b_h_new = tl.dot(b_k, b_d, allow_tf32=False)
        b_h_cumsum += b_h_new
        
        # Compute new
