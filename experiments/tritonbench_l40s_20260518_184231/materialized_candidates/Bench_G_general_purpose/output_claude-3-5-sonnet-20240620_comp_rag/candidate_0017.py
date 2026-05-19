import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                    start_m, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, 
                    BLOCK_N: tl.constexpr, STAGE: tl.constexpr, 
                    offs_m: tl.constexpr, offs_n: tl.constexpr, N_CTX: tl.constexpr):
    # Determine processing range based on stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:  # STAGE == 2
        lo = tl.multiple_of(start_m * BLOCK_M, BLOCK_M)
        hi = (start_m + 1) * BLOCK_M
        # Adjust pointers for stage 2
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo
    
    # Process blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load and scale K block
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM) < HEAD_DIM)[:, None]
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        
        # Compute attention scores
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        
        # Apply causal mask in stage 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        
        # Compute softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        
        # Update accumulators
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Load and accumulate V block
        v_mask = (offs_n[:, None] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM) < HEAD_DIM)[None, :]
        v = tl.load(V_ptrs, mask=v_mask)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        
        # Update pointers and maximums
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
        
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    # Get program ID and compute offsets
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute base offsets for Q, K, V
    qkv_offset = off_z * stride_qz + off_h * stride_qh
    scale_offset_q = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    scale_offset_k = off_hz * tl.cdiv(N_CTX, BLOCK_N)
    
    # Initialize pointers
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Setup pointers for Q, K, V blocks
    Q_ptrs = Q + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptrs = K + qkv_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptrs = V + qkv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Load Q block and scale
    q_mask = (offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(Q_ptrs, mask=q_mask)
    q_scale = tl.load(Q_scale + scale_offset_q + start_m)
    
    # Process attention in stages
    for stage in range(1, 3):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale, K_ptrs, K_scale + scale_offset_k, V_ptrs,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N, stage, offs_m, offs_n, N_CTX
        )
    
    # Normalize and store results
    acc = acc / l_i[:, None]
    out_ptrs = Out + qkv_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

def forward(q, k, v, q_scale, k_scale):
    """
    Wrapper function for the attention forward pass
    
    Args:
        q, k, v: Query, Key, and Value tensors
        q_scale, k_scale: Scaling factors for Q and K
    """
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Validate input dimensions
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Inconsistent feature dimensions"
    HEAD_DIM = q.shape[-1]
    
    # Prepare output tensor
    out = torch.empty_like(q)
    
    # Configure grid for kernel launch
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=2,
        num_warps=8
    )
    
    return out
