import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # CSR format indices and pointers
    row_indices, col_indices, values,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Strides for the tensors
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Block sizes (constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    # Program ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Initialize offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load row start and end for current block
    row_start = tl.load(row_indices + pid_block)
    row_end = tl.load(row_indices + pid_block + 1)

    # Compute base pointers
    q_base = Q + pid_batch * stride_qb + pid_head * stride_qh
    k_base = K + pid_batch * stride_kb + pid_head * stride_kh
    v_base = V + pid_batch * stride_vb + pid_head * stride_vh

    # Initialize softmax variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate through non-zero blocks
    for block_idx in range(row_start, row_end):
        col_block = tl.load(col_indices + block_idx)
        
        # Load Q block
        q_ptrs = q_base + (pid_block * BLOCK_M + offs_m[:, None]) * stride_qm + offs_d[None, :]
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)

        # Load K block
        k_ptrs = k_base + (col_block * BLOCK_N + offs_n[None, :]) * stride_kn + offs_d[:, None]
        k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len)

        # Compute attention scores
        qk = tl.dot(q, k) / (head_dim ** 0.5)
        
        # Compute softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update softmax stats
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Load V block and compute weighted sum
        v_ptrs = v_base + (col_block * BLOCK_N + offs_n[:, None]) * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len)

        # Scale attention weights
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        # Update accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        # Update running stats
        l_i = l_i_new
        m_i = m_i_new

    # Write output
    out_ptrs = Out + pid_batch * stride_ob + pid_head * stride_oh + \
               (pid_block * BLOCK_M + offs_m[:, None]) * stride_om + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)

def block_sparse_attention(q, k, v, row_indices, col_indices, values, block_size=64):
    """
    Wrapper function for block sparse attention.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        row_indices: CSR row pointers
        col_indices: CSR column indices
        values: CSR values
        block_size: Size of blocks for sparse computation
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Configure block sizes based on GPU capability
    BLOCK_M = BLOCK_N = block_size
    BLOCK_D = head_dim
    NUM_D_BLOCKS = triton.cdiv(head_dim, BLOCK_D)
    
    # Compute number of blocks
    num_blocks = row_indices.shape[0] - 1
    
    # Launch kernel
    grid = (batch_size, num_heads, num_blocks)
    
    block_sparse_attention_kernel[grid](
        q, k, v, output,
        row_indices, col_indices, values,
        batch_size, num_heads, seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
        num_warps=4,
    )
    
    return output
