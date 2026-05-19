import triton
import triton.language as tl
import torch
import math
from typing import Optional

@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    # Pointers to input/output tensors
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    # Strides for accessing tensors
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_hz, stride_hh,
    stride_oz, stride_oh, stride_on,
    # Block sizes and dimensions
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    H: tl.constexpr, N: tl.constexpr, 
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate indices
    num_blocks_m = tl.cdiv(BT, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    # Block indices
    bid_m = pid // num_blocks_n
    bid_n = pid % num_blocks_n
    
    # Starting indices for this block
    m_start = bid_m * BLOCK_M
    n_start = bid_n * BLOCK_N
    
    # Initialize offsets
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Create block pointers
    q_block_ptr = q_ptr + offs_m[:, None] * stride_qm
    k_block_ptr = k_ptr + offs_n[None, :] * stride_kn
    v_block_ptr = v_ptr + offs_n[:, None] * stride_vn
    h_block_ptr = h_ptr + offs_m[:, None] * stride_qm
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load and compute
    for h in range(0, H):
        # Load Q, K blocks
        q = tl.load(q_block_ptr + h * stride_qh)
        k = tl.load(k_block_ptr + h * stride_kh)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * scale
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Load V block and compute output
        v = tl.load(v_block_ptr + h * stride_vh)
        h_val = tl.load(h_block_ptr + h * stride_hh)
        
        # Accumulate weighted values
        acc += tl.dot(scores, v) * h_val
    
    # Write output
    offs_o = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptr + offs_o, acc)

def chunk_fwd_o_fn(q, k, v, h, scale=None):
    """
    Wrapper function to launch the Triton kernel
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Default scale if not provided
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Output tensor
    o = torch.empty_like(q)
    
    # Launch configs
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        h.stride(0), h.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        seq_len, head_dim, head_dim,  # BT, BK, BV
        num_heads, seq_len,
        scale,
        BLOCK_M, BLOCK_N
    )
    
    return o
