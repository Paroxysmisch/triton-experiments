import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BK': 64, 'BV': 64}, num_warps=8, num_stages=1),
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=1),
        triton.Config({'BK': 256, 'BV': 64}, num_warps=8, num_stages=1),
    ],
    key=['BT']
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    # Tensor strides
    s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t,
    s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
    # Problem dimensions and parameters
    scale, H, T,
    BT: tl.constexpr,  # Chunk size for tokens
    BK: tl.constexpr,  # Chunk size for K dimension
    BV: tl.constexpr,  # Chunk size for V dimension
):
    # Parallelize over batch-head and token chunks
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    
    # Compute batch and head indices
    bh_offset = pid_bh * (H * T)
    batch_head = pid_bh
    off_chunk = pid_chunk * BT
    
    # Create block pointers for Q, K, H
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_head * s_q_h,
        shape=(T, BK),
        strides=(s_q_t, 1),
        offsets=(off_chunk, 0),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_head * s_k_h,
        shape=(BK, T),
        strides=(1, s_k_t),
        offsets=(0, off_chunk),
        block_shape=(BK, BT),
        order=(0, 1)
    )
    h_block_ptr = tl.make_block_ptr(
        base=h_ptr + batch_head * s_h_h,
        shape=(T, BV),
        strides=(s_h_t, 1),
        offsets=(off_chunk, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    
    # Load blocks with boundary checks
    q = tl.load(q_block_ptr, boundary_check=(0,))
    k = tl.load(k_block_ptr, boundary_check=(1,))
    h = tl.load(h_block_ptr, boundary_check=(0,))
    
    # Compute S = QK^T * scale
    s = tl.dot(q, k, allow_tf32=False) * scale
    s = tl.exp(s)
    
    # Create causal mask if needed
    row_idx = tl.arange(0, BT) + off_chunk
    col_idx = tl.arange(0, BT) + off_chunk
    m_s = row_idx[:, None] >= col_idx[None, :]
    s = s * m_s
    
    # Load V and G blocks
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_head * s_v_h,
        shape=(T, BV),
        strides=(s_v_t, 1),
        offsets=(off_chunk, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    g_block_ptr = tl.make_block_ptr(
        base=g_ptr + batch_head * s_g_h,
        shape=(T, BV),
        strides=(s_g_t, 1),
        offsets=(off_chunk, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    v = tl.load(v_block_ptr, boundary_check=(0,))
    g = tl.load(g_block_ptr, boundary_check=(0,))
    
    # Compute output components
    b_o = tl.dot(s, v, allow_tf32=False)
    b_s = tl.sum(s, axis=1)
    
    # Update with gating mechanism
    o = (b_o + h * g) / b_s[:, None]
    
    # Store output block
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + batch_head * s_o_h,
        shape=(T, BV),
        strides=(s_o_t, 1),
        offsets=(off_chunk, 0),
        block_shape=(BT, BV),
        order=(1, 0)
    )
    tl.store(o_block_ptr, o, boundary_check=(0,))

### Wrapper Function
def chunk_fwd_o_fn(q, k, v, h, g, o, scale):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert h.is_cuda and g.is_cuda and o.is_cuda
    
    B, H, T, D = q.shape
    _, _, _, D_v = v.shape
    
    # Configure chunk sizes
    BT = 128 if T > 512 else 64  # Heuristic for chunk size
    BK = min(triton.next_power_of_2(D), 128)
    BV = min(triton.next_power_of_2(D_v), 128)
    
    grid = (B * H, triton.cdiv(T, BT))
    
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        q.stride(1), q.stride(2),  # s_q_h, s_q_t
        k.stride(1), k.stride(2),  # s_k_h, s_k_t
        v.stride(1), v.stride(2),  # s_v_h, s_v_t
        h.stride(1), h.stride(2),  # s_h_h, s_h_t
        g.stride(1), g.stride(2),  # s_g_h, s_g_t
        o.stride(1), o.stride(2),  # s_o_h, s_o_t
        scale, H, T,
        BT=BT, BK=BK, BV=BV
    )
