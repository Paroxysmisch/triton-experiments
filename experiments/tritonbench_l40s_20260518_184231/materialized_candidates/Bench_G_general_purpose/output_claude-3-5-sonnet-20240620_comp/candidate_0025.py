import torch
import triton
import triton.language as tl

# Constants for block sizes and other parameters
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    seq_len, num_heads, 
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    
    # Block position
    bid = pid // num_pid_n
    bid_n = pid % num_pid_n
    
    # Initialize pointers to Q, K, V
    q_start = Q + bid * stride_qm
    k_start = K + bid_n * stride_km
    v_start = V + bid_n * stride_vm
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    max_val = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    sum_val = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q block
    q_block_ptr = tl.make_block_ptr(
        q_start, 
        shape=(seq_len, BLOCK_D),
        strides=(stride_qd, 1),
        offsets=(bid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    q = tl.load(q_block_ptr)
    
    # Loop over K, V blocks
    for block_k in range(0, num_pid_n):
        # Load K block
        k_block_ptr = tl.make_block_ptr(
            k_start,
            shape=(seq_len, BLOCK_D),
            strides=(stride_kd, 1),
            offsets=(block_k * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0)
        )
        k = tl.load(k_block_ptr)
        
        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            row_idx = bid * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = block_k * BLOCK_N + tl.arange(0, BLOCK_N)
            causal_mask = col_idx[None, :] <= row_idx[:, None]
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Compute softmax
        max_prev = max_val
        max_val = tl.maximum(max_val, tl.max(qk, 1))
        exp_qk = tl.exp(qk - max_val[:, None])
        
        # Load V block
        v_block_ptr = tl.make_block_ptr(
            v_start,
            shape=(seq_len, BLOCK_D),
            strides=(stride_vd, 1),
            offsets=(block_k * BLOCK_N, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0)
        )
        v = tl.load(v_block_ptr)
        
        # Update accumulator
        sum_val = sum_val * tl.exp(max_prev - max_val) + tl.sum(exp_qk, 1)
        acc = acc * tl.exp(max_prev - max_val)[:, None] + tl.dot(exp_qk, v)
    
    # Compute final output
    out = acc / sum_val[:, None]
    
    # Write output
    out_block_ptr = tl.make_block_ptr(
        Out + bid * stride_om,
        shape=(seq_len, BLOCK_D),
        strides=(stride_od, 1),
        offsets=(bid * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0)
    )
    tl.store(out_block_ptr, out)

def flash_attn_triton(q, k, v, sm_scale, causal=False):
    """
    Compute Flash Attention.
    q, k, v: Query, Key, Value tensors of shape (batch_size, num_heads, seq_len, head_dim)
    sm_scale: Scale factor for dot products
    causal: Whether to apply causal masking
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Reshape input tensors
    q = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len, head_dim)
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Calculate strides
    stride_qm = q.stride(0)
    stride_qh = q.stride(1)
    stride_qd = q.stride(2)
    stride_km = k.stride(0)
    stride_kh = k.stride(1)
    stride_kd = k.stride(2)
    stride_vm = v.stride(0)
    stride_vh = v.stride(1)
    stride_vd = v.stride(2)
    stride_om = out.stride(0)
    stride_oh = out.stride(1)
    stride_od = out.stride(2)
    
    # Compute grid size
    grid = (triton.cdiv(seq_len, BLOCK_M) * batch_size * num_heads,)
    
    # Determine number of warps based on sequence length
    num_warps = 4
    if seq_len >= 2048:
        num_warps = 8
    elif seq_len >= 1024:
        num_warps = 4
    else:
        num_warps = 2
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale, out,
        stride_qm, stride_qh, stride_qd,
        stride_km, stride_kh, stride_kd,
        stride_vm, stride_vh, stride_vd,
        stride_om, stride_oh, stride_od,
        seq_len, num_heads,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=2,
    )
    
    # Reshape output back to original shape
    return out.reshape(batch_size, num_heads, seq_len, head_dim)
