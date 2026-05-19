import triton
import triton.language as tl
import torch

@triton.jit
def block_sparse_attention_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr,
    # CSR layout pointers
    layout_csr_row_indices_ptr, layout_csr_col_indices_ptr,
    # Dimensions
    batch_size, num_heads, num_kv_heads, seq_len, head_dim,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Other parameters
    softmax_scale,
    stride_qb, stride_qh, stride_qm,  # Strides for Q
    stride_kb, stride_kh, stride_kn,  # Strides for K
    stride_vb, stride_vh, stride_vn,  # Strides for V
    NUM_D_BLOCKS: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_query_blocks = tl.cdiv(seq_len, BLOCK_M)
    batch_head = pid // num_query_blocks
    query_block_idx = pid % num_query_blocks

    # Batch and head index calculation
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # Initialize row offsets for sparse blocks
    row_start = tl.load(layout_csr_row_indices_ptr + query_block_idx)
    row_end = tl.load(layout_csr_row_indices_ptr + query_block_idx + 1)

    # Initialize pointers for the query block
    q_block_ptr = q_ptr + (
        batch_idx * stride_qb +
        head_idx * stride_qh +
        query_block_idx * BLOCK_M * stride_qm
    )

    # Initialize accumulator for the output
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load query block
    q_block = tl.load(
        q_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qm +
        tl.arange(0, BLOCK_D)[None, :],
        mask=tl.arange(0, BLOCK_M)[:, None] < min(BLOCK_M, seq_len - query_block_idx * BLOCK_M),
        other=0.0
    )

    # Initialize max scores for softmax
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate through key blocks according to CSR layout
    for block_idx in range(row_start, row_end):
        col_idx = tl.load(layout_csr_col_indices_ptr + block_idx)
        
        # Load key block
        k_block_ptr = k_ptr + (
            batch_idx * stride_kb +
            kv_head_idx * stride_kh +
            col_idx * BLOCK_N * stride_kn
        )
        k_block = tl.load(
            k_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_kn +
            tl.arange(0, BLOCK_D)[None, :],
            mask=tl.arange(0, BLOCK_N)[:, None] < min(BLOCK_N, seq_len - col_idx * BLOCK_N),
            other=0.0
        )

        # Compute attention scores
        qk = tl.dot(q_block, k_block.transpose())
        qk = qk * softmax_scale

        # Update max scores and compute exponentials
        m_prev = m
        m = tl.maximum(m, tl.max(qk, 1))
        exp_qk = tl.exp(qk - m[:, None])
        
        # Update running sum
        l = l * tl.exp(m_prev - m) + tl.sum(exp_qk, 1)

        # Load value block
        v_block_ptr = v_ptr + (
            batch_idx * stride_vb +
            kv_head_idx * stride_vh +
            col_idx * BLOCK_N * stride_vn
        )
        v_block = tl.load(
            v_block_ptr + tl.arange(0, BLOCK_N)[:, None] * stride_vn +
            tl.arange(0, BLOCK_D)[None, :],
            mask=tl.arange(0, BLOCK_N)[:, None] < min(BLOCK_N, seq_len - col_idx * BLOCK_N),
            other=0.0
        )

        # Update accumulator
        acc += tl.dot(exp_qk, v_block)

    # Final rescaling
    acc = acc / l[:, None]

    # Write output
    out_block_ptr = out_ptr + (
        batch_idx * stride_qb +
        head_idx * stride_qh +
        query_block_idx * BLOCK_M * stride_qm
    )
    
    tl.store(
        out_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qm +
        tl.arange(0, BLOCK_D)[None, :],
        acc,
        mask=tl.arange(0, BLOCK_M)[:, None] < min(BLOCK_M, seq_len - query_block_idx * BLOCK_M)
    )

# Python wrapper function
def block_sparse_attention(q, k, v, layout_csr_row_indices, layout_csr_col_indices, 
                         num_heads, num_kv_heads, softmax_scale=None):
    """
    Compute block-sparse attention.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
        layout_csr_row_indices: CSR row indices for sparse block layout
        layout_csr_col_indices: CSR column indices for sparse block layout
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
    
    Returns:
        out: Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size = q.shape[0]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    
    # Set default softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Block sizes
    BLOCK_M = 64  # Query block size
    BLOCK_N = 64  # Key/Value block size
    BLOCK_D = min(32, head_dim)  # Head dimension block size
    NUM_D_BLOCKS = triton.cdiv(head_dim, BLOCK_D)
    
    # Compute number of blocks and grid
    num_query_blocks = triton.cdiv(seq_len, BLOCK_M)
    grid = (num_query_blocks * batch_size * num_heads,)
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Launch kernel
    block_sparse_attention_kernel[grid](
        q, k, v, out,
        layout_csr_row_indices, layout_csr_col_indices,
        batch_size, num_heads, num_kv_heads, seq_len, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        softmax_scale=softmax_scale,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2),
        stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2),
        stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2),
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    
    return out
