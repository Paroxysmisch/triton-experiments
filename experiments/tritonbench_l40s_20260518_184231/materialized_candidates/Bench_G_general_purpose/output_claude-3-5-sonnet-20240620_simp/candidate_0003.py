import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch index and head index
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Compute pointer offsets for current batch and head
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh

    # Load Q block
    Q_block_ptr = Q + q_offset
    K_block_ptr = K + k_offset
    V_block_ptr = V + v_offset
    
    # Initialize accumulator for attention scores
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Compute attention scores
    for d in range(0, head_dim, BLOCK_SIZE):
        # Load Q and K blocks
        q = tl.load(Q_block_ptr + d * stride_qd)
        k = tl.load(K_block_ptr + d * stride_kd)
        
        # Compute Q * K^T
        acc += tl.dot(q, tl.trans(k))
    
    # Scale attention scores
    scaling = 1.0 / math.sqrt(head_dim)
    acc *= scaling
    
    # Apply softmax
    acc = tl.softmax(acc)
    
    # Initialize output accumulator
    out_acc = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    
    # Compute attention * V
    for d in range(0, seq_len, BLOCK_SIZE):
        v = tl.load(V_block_ptr + d * stride_vs)
        out_acc += tl.dot(acc, v)
    
    # Store result
    Out_block_ptr = Out + o_offset
    tl.store(Out_block_ptr, out_acc)

def context_attention_fwd(q, k, v):
    """
    Forward pass for context attention mechanism.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Allocate output tensor
    output = torch.empty_like(q)
    
    # Configure meta-parameters
    BLOCK_SIZE = 32
    
    # Configure grid
    grid = (batch_size * num_heads,)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, output,
        batch_size, num_heads, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
