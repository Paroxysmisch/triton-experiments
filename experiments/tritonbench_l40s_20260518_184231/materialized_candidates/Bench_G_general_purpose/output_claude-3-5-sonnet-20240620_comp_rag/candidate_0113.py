import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,           # Accumulator and softmax intermediates 
    q, q_scale,              # Query matrix and scale
    K_ptrs, K_scale_ptr,     # Key matrix pointers and scale
    V_ptrs,                  # Value matrix pointers
    start_m,                 # Starting block index
    BLOCK_M: tl.constexpr,   # Block sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,     # Processing stage
    offs_m: tl.constexpr,    # Offset arrays
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr      # Context size
):
    # Define processing bounds based on stage
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
        
        # Load and scale key block
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM) < HEAD_DIM)[:, None]
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        
        # Compute attention scores
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale

        # Apply causal mask in stage 2
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            
        # Update running max for numerical stability
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        
        # Compute softmax values
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        # Update accumulators
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Load and apply values
        v_mask = (offs_n[:, None] < (N_CTX - start_n)) & (tl.arange(0, HEAD_DIM)[None, :] < HEAD_DIM)
        v = tl.load(V_ptrs, mask=v_mask)
        
        # Accumulate weighted values
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16))
        
        # Update state
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
        
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V,                # Input matrices
    Q_scale, K_scale,       # Scale factors
    Out,                    # Output matrix
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K
    stride_vz, stride_vh, stride_vk, stride_vn,  # Strides for V
    stride_oz, stride_oh, stride_om, stride_on,  # Strides for output
    Z, H, N_CTX,           # Batch, heads and context size
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr
):
    # Program ID gives starting block and head/batch position
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Calculate offsets
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize pointers and offsets
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Calculate matrix pointers
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Load query block
    q_mask = (offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM)
    q = tl.load(Q_ptrs, mask=q_mask)
    q_scale = tl.load(Q_scale + off_hz * tl.cdiv(N_CTX, BLOCK_M) + start_m)
    
    # Process attention in stages
    for stage in range(STAGE, 0, -1):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, q_scale,
            K_ptrs, K_scale + off_hz * tl.cdiv(N_CTX, BLOCK_N), V_ptrs,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
            stage, offs_m, offs_n, N_CTX
        )
    
    # Normalize and store results
    acc = acc / l_i[:, None]
    O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=q_mask)
