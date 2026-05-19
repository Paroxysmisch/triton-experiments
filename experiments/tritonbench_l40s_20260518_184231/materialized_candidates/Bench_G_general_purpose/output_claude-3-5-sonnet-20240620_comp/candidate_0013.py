import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Matrix dimensions
    Lq, Lk, Lv, H, D,
    # Strides for accessing tensors
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    # Scale for attention scores
    sm_scale,
    # Block sizes for tiling
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    # Number of KV groups for grouped attention
    kv_group_num: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Lq, BLOCK_M)
    num_pid_n = tl.cdiv(Lk, BLOCK_N)
    
    # Block ID and offsets
    bid = pid // num_pid_n
    nid = pid % num_pid_n
    
    # Initialize offsets
    start_m = bid * BLOCK_M
    start_n = nid * BLOCK_N
    
    # Initialize pointers
    offs_q = start_m + tl.arange(0, BLOCK_M)
    offs_k = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load query block
    q = tl.load(Q + offs_q * stride_qm, 
                mask=offs_q < Lq, other=0.0)
    
    # Load key block
    k = tl.load(K + offs_k * stride_kn,
                mask=offs_k < Lk, other=0.0)
    
    # Compute attention scores
    scores = tl.dot(q, k.T) * sm_scale
    
    # Apply causal mask
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    scores = tl.where(causal_mask, scores, float("-inf"))
    
    # Compute softmax
    scores = scores - tl.max(scores, axis=1)[:, None]
    scores = tl.exp(scores)
    scores = scores / tl.sum(scores, axis=1)[:, None]
    
    # Load values
    v = tl.load(V + offs_k * stride_vn,
                mask=offs_k < Lv, other=0.0)
    
    # Compute weighted sum
    out = tl.dot(scores, v)
    
    # Store output
    offs_out = start_m + tl.arange(0, BLOCK_M)
    tl.store(Out + offs_out * stride_on,
             out, mask=offs_out < Lq)

def context_attention_fwd(q, k, v, use_tesla=False):
    # Shape constraints
    batch_size, num_heads, seq_len, d_head = q.shape
    assert k.shape[3] == d_head and v.shape[3] == d_head
    
    # Compute optimal block size
    BLOCK_M = 128 if use_tesla else 256
    BLOCK_N = 64
    
    # Compute scaling factor
    sm_scale = 1.0 / math.sqrt(d_head)
    
    # Prepare output tensor
    out = torch.empty_like(q)
    
    # Calculate strides
    stride_qb = q.stride(0)
    stride_qh = q.stride(1)
    stride_qm = q.stride(2)
    stride_kb = k.stride(0)
    stride_kh = k.stride(1)
    stride_kn = k.stride(2)
    stride_vb = v.stride(0)
    stride_vh = v.stride(1)
    stride_vn = v.stride(2)
    stride_ob = out.stride(0)
    stride_oh = out.stride(1)
    stride_on = out.stride(2)
    
    # Configure grid
    grid = lambda meta: (
        triton.cdiv(seq_len, meta['BLOCK_M']) * 
        triton.cdiv(seq_len, meta['BLOCK_N']),
    )
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, out,
        seq_len, seq_len, seq_len, num_heads, d_head,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_ob, stride_oh, stride_on,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        kv_group_num=1,
    )
    
    return out
