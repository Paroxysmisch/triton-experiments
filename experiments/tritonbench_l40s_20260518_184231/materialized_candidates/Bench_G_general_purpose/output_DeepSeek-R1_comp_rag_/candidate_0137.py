import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    softmax_scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    # Grid dimensions
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    # Head grouping (GQA)
    kv_head_id = head_id // (num_heads // num_kv_heads)
    
    # Load CSR row pointers
    row_start = tl.load(layout_csr_row_indices + q_block_id)
    row_end = tl.load(layout_csr_row_indices + q_block_id + 1)
    
    # Calculate offsets for Q block
    q_offset = (
        batch_id * stride_qb + 
        head_id * stride_qh + 
        q_block_id * BLOCK_M * stride_qd
    )
    q_ptrs = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qd + tl.arange(0, BLOCK_D)[None, :]
    
    # Load Q block with masking
    q = tl.load(q_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D), other=0.0)
    
    # Initialize accumulation registers
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate over sparse columns using CSR data
    for col_idx in range(row_start, row_end):
        k_block_id = tl.load(layout_csr_col_indices + col_idx)
        
        # Load K block
        k_offset = (
            batch_id * stride_kb +
            kv_head_id * stride_kh +
            k_block_id * BLOCK_N * stride_kd
        )
        k_ptrs = K + k_offset + tl.arange(0, BLOCK_D)[:, None] * stride_kd + tl.arange(0, BLOCK_N)[None, :]
        k = tl.load(k_ptrs, mask=(tl.arange(0, BLOCK_D)[:, None] < BLOCK_D) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N), other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k, allow_tf32=False)
        qk *= softmax_scale
        
        # Load V block
        v_offset = (
            batch_id * stride_vb +
            kv_head_id * stride_vh +
            k_block_id * BLOCK_N * stride_vd
        )
        v_ptrs = V + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vd + tl.arange(0, BLOCK_D)[None, :]
        v = tl.load(v_ptrs, mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D), other=0.0)
        
        # Stable softmax computation
        m_curr = tl.maximum(tl.max(qk, axis=1), m_max)
        alpha = tl.exp(m_max - m_curr)
        beta = tl.exp(qk - m_curr[:, None])
        
        l_curr = alpha * l_sum + tl.sum(beta, axis=1)
        p = beta / l_curr[:, None]
        
        # Update accumulators
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_max = m_curr
        l_sum = l_curr

    # Handle multiple D blocks if needed
    if NUM_D_BLOCKS > 1:
        for d_block in range(1, NUM_D_BLOCKS):
            acc += ...  # Additional accumulation logic

    # Store final result
    out_offset = (
        batch_id * stride_ob +
        head_id * stride_oh +
        q_block_id * BLOCK_M * stride_od
    )
    out_ptrs = Out + out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_od + tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptrs, acc, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D))

def block_sparse_attention_forward(
    q, k, v, 
    layout_csr_row_indices,
    layout_csr_col_indices,
    output,
    num_heads: int,
    num_kv_heads: int,
    softmax_scale: float = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64
):
    # Auto-configure settings based on hardware
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        BLOCK_M, BLOCK_N = 128, 128
    
    # Derive parameters
    Lk = q.size(-1)
    batch_size = q.size(0)
    num_query_blocks = layout_csr_row_indices.size(0) - 1
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (Lk ** 0.5)
    
    # Configure grid
    grid = (batch_size, num_heads, num_query_blocks)
    
    # Launch kernel
    block_sparse_attention_kernel[grid](
        q, k, v,
        layout_csr_row_indices,
        layout_csr_col_indices,
        output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        softmax_scale=softmax_scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=1 if Lk <= BLOCK_D else Lk // BLOCK_D,
        num_warps=4 if BLOCK_D <= 64 else 8,
        num_stages=3
    )
