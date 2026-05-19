import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch and head idx
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    batch_idx = pid // (num_block_m * H)
    head_idx = (pid % (num_block_m * H)) // num_block_m
    block_idx = pid % num_block_m

    # Initialize offsets
    offs_m = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + batch_idx * stride_kz + head_idx * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + batch_idx * stride_vz + head_idx * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    sum_score = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Loop over K,V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K,V blocks
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q, k) * sm_scale
        
        # Apply causal mask if needed
        if CAUSAL:
            scores = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), scores, float("-inf"))
        
        # Apply softmax
        scores = tl.softmax(scores, axis=1)
        
        # Compute attention output
        acc += tl.dot(scores, v)
    
    # Store output
    out_ptrs = Out + batch_idx * stride_oz + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def flash_attn_triton(q, k, v, causal=False):
    """
    Compute Flash Attention using Triton.
    
    Args:
        q: Query tensor of shape (batch_size, n_heads, seq_len, d_head)
        k: Key tensor of shape (batch_size, n_heads, seq_len, d_head)
        v: Value tensor of shape (batch_size, n_heads, seq_len, d_head)
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor of shape (batch_size, n_heads, seq_len, d_head)
    """
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Initialize output
    o = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = d_head
    
    # Compute scale for attention scores
    sm_scale = 1.0 / (d_head ** 0.5)
    
    # Launch kernel
    grid = (batch_size * n_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    _fwd_kernel[grid](
        q, k, v, sm_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        CAUSAL=causal,
    )
    
    return o
