import torch
import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, g_ptr, h_ptr, do_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    # Matrix dimensions
    batch_size, n_heads, seq_len, d_head,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_vb, stride_vh, stride_vt,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(batch_size, BLOCK_SIZE)
    num_pid_h = tl.cdiv(n_heads, BLOCK_HEAD)
    
    # Block indices
    bid = pid // (num_pid_h * seq_len)
    bhid = (pid % (num_pid_h * seq_len)) // seq_len
    tid = pid % seq_len

    # Initialize offsets
    offs_b = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_h = bhid * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, d_head)
    
    # Compute batch/head/sequence offsets
    q_offset = offs_b[:, None, None] * stride_qb + offs_h[None, :, None] * stride_qh + tid * stride_qt
    k_offset = offs_b[:, None, None] * stride_kb + offs_h[None, :, None] * stride_kh + tid * stride_kt
    v_offset = offs_b[:, None, None] * stride_vb + offs_h[None, :, None] * stride_vh + tid * stride_vt
    
    # Load inputs
    q = tl.load(q_ptr + q_offset)
    k = tl.load(k_ptr + k_offset)
    v = tl.load(v_ptr + v_offset)
    g = tl.load(g_ptr + q_offset)
    h = tl.load(h_ptr + q_offset)
    do = tl.load(do_ptr + q_offset)
    
    # Compute gradients
    scale = 1.0 / tl.sqrt(d_head)
    
    # Gradient for Q
    b_dq = tl.zeros([BLOCK_SIZE, BLOCK_HEAD, d_head], dtype=tl.float32)
    for i in range(seq_len):
        k_i = tl.load(k_ptr + k_offset + i * stride_kt)
        v_i = tl.load(v_ptr + v_offset + i * stride_vt)
        do_i = tl.load(do_ptr + q_offset + i * stride_qt)
        
        qk = tl.dot(q, k_i.T) * scale
        b_dq += tl.dot(do_i * g, v_i) * qk
    
    # Gradient for K
    b_dk = tl.zeros([BLOCK_SIZE, BLOCK_HEAD, d_head], dtype=tl.float32)
    for i in range(seq_len):
        q_i = tl.load(q_ptr + q_offset + i * stride_qt)
        v_i = tl.load(v_ptr + v_offset + i * stride_vt)
        do_i = tl.load(do_ptr + q_offset + i * stride_qt)
        g_i = tl.load(g_ptr + q_offset + i * stride_qt)
        
        qk = tl.dot(q_i, k.T) * scale
        b_dk += tl.dot((do_i * g_i).T, v_i) * qk
    
    # Gradient for G
    b_dg = tl.zeros([BLOCK_SIZE, BLOCK_HEAD, d_head], dtype=tl.float32)
    for i in range(seq_len):
        h_i = tl.load(h_ptr + q_offset + i * stride_qt)
        do_i = tl.load(do_ptr + q_offset + i * stride_qt)
        b_dg += do_i * h_i
    
    # Store results
    tl.store(dq_ptr + q_offset, b_dq)
    tl.store(dk_ptr + k_offset, b_dk)
    tl.store(dg_ptr + q_offset, b_dg)

# Wrapper function
def chunk_bwd_dqkg_fn(q, k, v, g, h, do):
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Allocate output tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    
    # Configure kernel parameters
    BLOCK_SIZE = 8
    BLOCK_HEAD = 8
    
    # Define grid
    grid = (batch_size * n_heads * seq_len,)
    
    # Launch kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, g, h, do,
        dq, dk, dg,
        batch_size, n_heads, seq_len, d_head,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_HEAD=BLOCK_HEAD
    )
    
    return dq, dk, dg
