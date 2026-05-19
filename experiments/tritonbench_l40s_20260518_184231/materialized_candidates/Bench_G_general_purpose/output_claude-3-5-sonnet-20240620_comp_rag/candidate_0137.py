import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    # Pointers to tensors
    Q, K, V, Out,
    # CSR layout information
    layout_csr_row_indices, layout_csr_col_indices,
    # Strides and shapes
    stride_qbs, stride_qh, stride_kbs, stride_kh,
    stride_vbs, stride_vh, stride_obs, stride_oh,
    # Constants
    softmax_scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    # Get program ID
    query_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    # Calculate batch and head indices
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Get row start and end indices from CSR format
    row_start = tl.load(layout_csr_row_indices + query_block_idx)
    row_end = tl.load(layout_csr_row_indices + query_block_idx + 1)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load query block
    q_offset = (
        batch_idx * stride_qbs +
        head_idx * stride_qh +
        query_block_idx * BLOCK_M * BLOCK_D
    )
    q = tl.load(Q + q_offset + offs_m[:, None] * BLOCK_D + offs_d[None, :])
    
    # Iterate over key blocks according to CSR layout
    for block_idx in range(row_start, row_end):
        key_block_idx = tl.load(layout_csr_col_indices + block_idx)
        
        # Load key block
        k_offset = (
            batch_idx * stride_kbs +
            kv_head_idx * stride_kh +
            key_block_idx * BLOCK_N * BLOCK_D
        )
        k = tl.load(K + k_offset + offs_n[None, :] * BLOCK_D + offs_d[:, None])
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = qk * softmax_scale
        
        # Compute softmax values
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update running statistics
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # Load value block and compute attention
        v_offset = (
            batch_idx * stride_vbs +
            kv_head_idx * stride_vh +
            key_block_idx * BLOCK_N * BLOCK_D
        )
        v = tl.load(V + v_offset + offs_n[:, None] * BLOCK_D + offs_d[None, :])
        
        # Update accumulator
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Store output
    out_offset = (
        batch_idx * stride_obs +
        head_idx * stride_oh +
        query_block_idx * BLOCK_M * BLOCK_D
    )
    tl.store(Out + out_offset + offs_m[:, None] * BLOCK_D + offs_d[None, :], acc)

def block_sparse_attention(q, k, v, layout_csr_row_indices, layout_csr_col_indices, num_heads, num_kv_heads):
    """
    Wrapper function for block sparse attention kernel
    """
    batch_size = q.shape[0]
    num_blocks = layout_csr_row_indices.shape[0] - 1
    block_size_m = 64  # Can be tuned based on GPU architecture
    block_size_n = 64
    block_size_d = q.shape[-1]
    
    # Compute softmax scaling factor
    softmax_scale = 1.0 / (block_size_d ** 0.5)
    
    # Allocate output tensor
    output = torch.empty_like(q)
    
    # Configure grid and launch kernel
    grid = (num_blocks, batch_size * num_heads)
    
    block_sparse_attention_kernel[grid](
        q, k, v, output,
        layout_csr_row_indices, layout_csr_col_indices,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        softmax_scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_D=block_size_d,
        NUM_D_BLOCKS=1,
        num_warps=4,
    )
    
    return output
