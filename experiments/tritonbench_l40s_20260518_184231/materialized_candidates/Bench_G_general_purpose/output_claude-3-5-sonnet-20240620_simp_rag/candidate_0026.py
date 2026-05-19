import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,                   # Accumulators
    q, q_scale,                      # Query and its scale
    K_ptrs, K_scale_ptr, V_ptrs,     # Key/Value pointers and key scale
    start_m,                         # Starting position
    BLOCK_M: tl.constexpr,          # Static params
    HEAD_DIM: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr
):
    # Determine processing range based on stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:  # STAGE == 2
        lo = tl.multiple_of(start_m * BLOCK_M, BLOCK_M)
        hi = (start_m + 1) * BLOCK_M
        # Adjust pointers based on offset
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo

    # Process blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load and scale keys
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & ((tl.arange(0, 128) < 96)[:, None])
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        
        # Compute attention scores
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        
        # Apply causal mask in stage 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        # Compute softmax and update accumulators
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load values and update accumulator
        v_mask = (offs_n[:, None] < (N_CTX - start_n)) & ((tl.arange(0, 128) < 96)[None, :])
        v = tl.load(V_ptrs, mask=v_mask)
        p = p.to(tl.float16)
        acc += tl.dot(p, v.to(tl.float16), out_dtype=tl.float16)
        
        # Update pointers and maximum
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
        
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,  # Input/output tensors
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,                      # Dimensions
    HEAD_DIM: tl.constexpr,           # Static params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    # Get program ID and offsets
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Calculate base offsets
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)

    # Initialize offsets for blocks
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)

    # Calculate pointers
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    # Load query and scale
    q_mask = (offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :])
    q = tl.load(Q_ptrs, mask=q_mask)
    q_scale = tl.load(Q_scale_ptr)

    # Process attention in stages
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX
    )
    
    acc, l_i, _ = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX
    )

    # Normalize and store result
    acc = acc / l_i[:, None]
    out_mask = (offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :])
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=out_mask)

def forward(q, k, v, q_scale, k_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V
    
    # Initialize output tensor
    o = torch.empty_like(q, dtype=torch.bfloat16)
    
    # Calculate grid dimensions
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], N_CTX=q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        STAGE=3, num_warps=8, num_stages=3
    )
    
    return o
