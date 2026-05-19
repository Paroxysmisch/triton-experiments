import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i,
    q, k, v,
    q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    
    # Block indices
    bid_zm = pid // (num_block_m * H)  # Batch index
    bhm = (pid % (num_block_m * H))
    bid_h = bhm // num_block_m         # Head index
    bid_m = bhm % num_block_m          # Row block index
    
    # Initialize pointers to Q, K, V
    q_ptr = q + bid_zm * stride_qz + bid_h * stride_qh + bid_m * BLOCK_M * stride_qm
    k_ptr = k + bid_zm * stride_kz + bid_h * stride_kh
    v_ptr = v + bid_zm * stride_vz + bid_h * stride_vh
    
    # Initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load Q block
    q_block = tl.load(q_ptr + offs_m[:, None] * stride_qm + 
                      tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk,
                      mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Scale Q
    q_block = q_block * q_scale
    
    # Loop over K,V blocks
    for n in range(0, N_CTX, BLOCK_N):
        # Load K,V blocks
        k_block = tl.load(k_ptr + n * stride_kn + 
                         offs_n[:, None] * stride_kn + 
                         tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk,
                         mask=offs_n[:, None] < (N_CTX - n), other=0.0)
        v_block = tl.load(v_ptr + n * stride_vn + 
                         offs_n[:, None] * stride_vn + 
                         tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk,
                         mask=offs_n[:, None] < (N_CTX - n), other=0.0)
        
        # Scale K
        k_block = k_block * k_scale
        
        # Compute attention scores
        scores = tl.dot(q_block, tl.trans(k_block))
        scores = tl.exp(scores)
        
        # Update running sum
        l_i += tl.sum(scores, axis=1)
        
        # Update accumulator
        acc += tl.dot(scores, v_block)

@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    Q_scale, K_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Initialize accumulator and normalizing factor
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Call inner kernel
    _attn_fwd_inner(
        acc, l_i, Q, K, V, Q_scale, K_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        Z, H, N_CTX, BLOCK_M, BLOCK_N, BLOCK_DMODEL
    )
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Write output
    pid = tl.program_id(0)
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    bid_zm = pid // (num_block_m * H)
    bhm = (pid % (num_block_m * H))
    bid_h = bhm // num_block_m
    bid_m = bhm % num_block_m
    
    offs_m = tl.arange(0, BLOCK_M)
    out_ptr = Out + bid_zm * stride_oz + bid_h * stride_oh + bid_m * BLOCK_M * stride_om
    
    tl.store(out_ptr + offs_m[:, None] * stride_om + 
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_ok,
             acc, mask=offs_m[:, None] < N_CTX)

# PyTorch wrapper
def attention_forward(q, k, v, q_scale, k_scale):
    """
    q: (batch_size, num_heads, seq_len, d_model)
    k: (batch_size, num_heads, seq_len, d_model)
    v: (batch_size, num_heads, seq_len, d_model)
    """
    batch_size, num_heads, seq_len, d_model = q.shape
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = d_model
    
    # Launch kernel
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    _attn_fwd[grid](
        q, k, v, out,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out
