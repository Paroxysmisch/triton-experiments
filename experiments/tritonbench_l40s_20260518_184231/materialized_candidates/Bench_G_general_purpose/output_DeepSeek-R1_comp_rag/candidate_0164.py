import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_ptrs, V_ptrs,  # Pointers for K and V tensors
    start_m,  # Starting index for the current block
    scale,  # Scaling factor for the dot product (1 / sqrt(HEAD_DIM))
    BLOCK_M: tl.constexpr,  # Block size for queries
    HEAD_DIM: tl.constexpr,  # Dimension of each head
    BLOCK_N: tl.constexpr,  # Block size for keys/values
    IS_CAUSAL: tl.constexpr,  # Whether to apply causal masking
    offs_m: tl.constexpr,  # Offsets for the current query block
    offs_n: tl.constexpr,  # Offsets for the key/value block
    N_CTX: tl.constexpr,  # Total sequence length
):
    # Loop over blocks of keys/values
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_ptrs + start_n * HEAD_DIM, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale  # Scale the dot product
        
        # Apply causal masking if required
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = qk * causal_mask + (1 - causal_mask) * -1e6
        
        # Compute softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)
        
        # Update accumulators
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Load value block and accumulate
        v = tl.load(V_ptrs + start_n * HEAD_DIM, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update max and sum for numerical stability
        m_i = m_ij
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, Out,  # Tensor pointers
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K
    stride_vz, stride_vh, stride_vn, stride_vk,  # Strides for V
    stride_oz, stride_oh, stride_om, stride_ok,  # Strides for Out
    Z, H, N_CTX,  # Shape parameters: batch size, heads, seq length
    HEAD_DIM: tl.constexpr,  # Dimension per head
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys/values
    IS_CAUSAL: tl.constexpr,  # Apply causal masking
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Compute pointers for the current batch and head
    q_offset = off_z * stride_qz + off_h * stride_qh + start_m * BLOCK_M * stride_qm
    Q_ptr = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    K_ptr = K + off_z * stride_kz + off_h * stride_kh + tl.arange(0, BLOCK_N)[None, :] * stride_kn + tl.arange(0, HEAD_DIM)[:, None] * stride_kk
    V_ptr = V + off_z * stride_vz + off_h * stride_vh + tl.arange(0, BLOCK_N)[:, None] * stride_vn + tl.arange(0, HEAD_DIM)[None, :] * stride_vk
    O_ptr = Out + off_z * stride_oz + off_h * stride_oh + start_m * BLOCK_M * stride_om + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Load query block
    q = tl.load(Q_ptr, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < N_CTX, other=0.0)
    scale = 1.0 / tl.sqrt(tl.float32(HEAD_DIM))  # Scaling factor
    
    # Process inner blocks
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, K_ptr, V_ptr,
        start_m, scale, BLOCK_M, HEAD_DIM, BLOCK_N, IS_CAUSAL,
        tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_N), N_CTX
    )
    
    # Normalize and store output
    acc = acc / l_i[:, None]
    tl.store(O_ptr, acc.to(Out.type.element_ty), mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < N_CTX)

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        # Ensure inputs are contiguous
        q, k, v = [x.contiguous() for x in (q, k, v)]
        batch_size, n_heads, seq_len, d_head = q.shape
        
        # Initialize output tensor
        o = torch.empty_like(q)
        
        # Configure kernel launch parameters
        BLOCK_M = 128
        BLOCK_N = 64 if d_head <= 64 else 128
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_heads)
        
        # Launch kernel
        _attn_fwd[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch_size, n_heads, seq_len,
            HEAD_DIM=d_head,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=4,
        )
        return o

def attention(q, k, v, is_causal=False):
    return AttentionFunction.apply(q, k, v, is_causal)
