import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,                   # Accumulators for attention computation
    q, q_scale,                      # Query matrix and its scale
    K_ptrs, K_scale_ptr, V_ptrs,     # Key/Value pointers and key scale pointer
    start_m,                         # Starting position for block processing
    BLOCK_M: tl.constexpr,          # Block size for M dimension
    HEAD_DIM: tl.constexpr,         # Dimension of attention heads
    BLOCK_N: tl.constexpr,          # Block size for N dimension
    STAGE: tl.constexpr,            # Processing stage
    offs_m: tl.constexpr,           # M dimension offsets
    offs_n: tl.constexpr,           # N dimension offsets
    N_CTX: tl.constexpr            # Context size
):
    # Determine processing range based on stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    else:  # STAGE == 2
        lo = tl.multiple_of(start_m * BLOCK_M, BLOCK_M)
        hi = (start_m + 1) * BLOCK_M
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo

    # Process blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load and process key block
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

        # Load and process value block
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < (N_CTX - start_n)) & 
                   ((tl.arange(0, 128) < 96)[None, :]))
        
        # Update accumulator with attention-weighted values
        p = p.to(tl.float16)
        acc += tl.dot(p, v.to(tl.float16), out_dtype=tl.float16)
        
        # Update pointers and maximum values
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM

    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,  # Input/output tensors
    stride_qz, stride_qh, stride_qm, stride_qk,  # Query strides
    stride_kz, stride_kh, stride_kn, stride_kk,  # Key strides
    stride_vz, stride_vh, stride_vk, stride_vn,  # Value strides
    stride_oz, stride_oh, stride_om, stride_on,  # Output strides
    Z, H, N_CTX,                      # Batch size, heads, context length
    HEAD_DIM: tl.constexpr,           # Head dimension
    BLOCK_M: tl.constexpr,            # Block size M
    BLOCK_N: tl.constexpr,            # Block size N
    STAGE: tl.constexpr               # Processing stage
):
    # Calculate program IDs and offsets
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Calculate base offsets for tensors
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)

    # Initialize offsets for block processing
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)

    # Calculate pointers for all tensors
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

    # Load query block
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :]))
    q_scale = tl.load(Q_scale_ptr)

    # Process attention in two stages
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX
    )
    
    acc, l_i, _ = _attn_fwd_inner(
        acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX
    )

    # Normalize and store results
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), 
             mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :]))
