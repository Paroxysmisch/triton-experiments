import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,  # Pointers to matrices
    stride_qb, stride_qh, stride_qm,  # Strides for Q matrix
    stride_kb, stride_kh, stride_kn,  # Strides for K matrix 
    stride_vb, stride_vh, stride_vn,  # Strides for V matrix
    stride_ob, stride_oh, stride_om,  # Strides for output matrix
    nheads, seqlen_q, seqlen_k, headdim,  # Matrix dimensions
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  # Block sizes
    BLOCK_HEADDIM: tl.constexpr,  # Head dimension block size
    causal: tl.constexpr  # Causal mask flag
):
    # Program ID gives us the block we're computing
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = tl.cdiv(seqlen_k, BLOCK_N)
    
    # Initialize pointers to the Q matrix
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    
    # Load Q block
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    
    # Iterate through K,V blocks
    for start_n in range(0, seqlen_k, BLOCK_N):
        # Load K block
        k_ptrs = K + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k)
        
        # Load V block
        v_ptrs = V + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k)
        
        # Compute attention scores
        scores = tl.dot(q, k, trans_b=True)
        
        # Apply causal mask if needed
        if causal:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            scores = scores * causal_mask
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Compute output
        acc += tl.dot(scores.to(v.dtype), v)
    
    # Store output
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seqlen_q)

def flash_attn_triton(q, k, v, causal=False):
    """
    Compute Flash Attention.
    q: (batch_size, nheads, seqlen_q, headdim)
    k: (batch_size, nheads, seqlen_k, headdim)
    v: (batch_size, nheads, seqlen_k, headdim)
    """
    batch_size, nheads, seqlen_q, headdim = q.shape
    _, _, seqlen_k, _ = k.shape
    
    # Ensure contiguous inputs
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Initialize output
    o = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = min(128, triton.next_power_of_2(headdim))
    
    # Launch kernel
    grid = (triton.cdiv(seqlen_q, BLOCK_M),)
    _fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, seqlen_q, seqlen_k, headdim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        causal=causal
    )
    
    return o
