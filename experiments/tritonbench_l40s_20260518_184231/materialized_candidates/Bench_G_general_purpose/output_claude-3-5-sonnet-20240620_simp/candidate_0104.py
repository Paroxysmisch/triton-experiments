import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, q, k, v, q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Pointers for the current block
    q_ptrs = q + (stride_qz * tl.arange(0, BLOCK_M)) 
    k_ptrs = k + (stride_kz * tl.arange(0, BLOCK_N))
    v_ptrs = v + (stride_vz * tl.arange(0, BLOCK_N))
    
    # Load query and key blocks
    q_block = tl.load(q_ptrs)
    k_block = tl.load(k_ptrs)
    
    # Scale Q and K
    q_block = q_block * q_scale
    k_block = k_block * k_scale
    
    # Compute attention scores
    scores = tl.dot(q_block, k_block.transpose())
    scores = scores * (1.0 / tl.sqrt(float(BLOCK_DMODEL)))
    
    # Apply softmax
    scores = tl.softmax(scores)
    
    # Load value block and compute weighted sum
    v_block = tl.load(v_ptrs)
    acc += tl.dot(scores, v_block)
    l_i += tl.sum(scores, axis=1)
    
    return acc, l_i

@triton.jit
def _attn_fwd(
    q, k, v, o,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, q_scale, k_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Initialize accumulator and normalization factor
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Main loop over N_CTX dimension
    for n in range(0, N_CTX, BLOCK_N):
        acc, l_i = _attn_fwd_inner(
            acc, l_i, q, k, v, q_scale, k_scale,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            Z, H, N_CTX,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL
        )
    
    # Store output
    o_ptrs = o + (pid * stride_oz + tl.arange(0, BLOCK_M) * stride_om)
    tl.store(o_ptrs, acc)

def forward(q, k, v, q_scale, k_scale):
    """
    Forward pass of attention mechanism
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, d_model)
        k: Key tensor of shape (batch_size, num_heads, seq_len, d_model)
        v: Value tensor of shape (batch_size, num_heads, seq_len, d_model)
        q_scale: Scaling factor for query
        k_scale: Scaling factor for key
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, d_model)
    """
    batch_size, num_heads, seq_len, d_model = q.shape
    
    # Output tensor
    o = torch.empty_like(q)
    
    # Grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_DMODEL = d_model
    
    # Launch kernel
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    _attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, num_heads, seq_len,
        q_scale, k_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
    
    return o
