import torch
import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr,
    layout_csr_row_ptr, layout_csr_col_ptr,
    out_ptr,
    # Strides for Q tensor
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_d,
    # Strides for K tensor
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_d,
    # Strides for V tensor
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_d,
    # Strides for output tensor
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_d,
    # Static parameters
    softmax_scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    # Get program IDs
    m_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_size = tl.num_programs(1) // num_heads
    batch_idx = batch_head // num_heads
    head_idx = batch_head % num_heads
    
    # Compute grouped attention head index
    G = num_heads // num_kv_heads
    kv_head_idx = head_idx // G
    
    # Load CSR row indices for current block
    start_row = tl.load(layout_csr_row_ptr + m_block)
    end_row = tl.load(layout_csr_row_ptr + m_block + 1)
    num_n_blocks = end_row - start_row
    
    # Calculate Q offset
    q_seq_offset = m_block * BLOCK_M
    q_offset = (batch_idx * stride_q_batch + 
                head_idx * stride_q_head + 
                q_seq_offset * stride_q_seq)
    
    # Initialize QK accumulator
    qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over D blocks for QK^T accumulation
    for d_block in range(NUM_D_BLOCKS):
        # Load Q block
        q_d_offset = q_offset + d_block * BLOCK_D * stride_q_d
        q = tl.load(
            q_ptr + q_d_offset + 
            tl.arange(0, BLOCK_M)[:, None] * stride_q_seq + 
            tl.arange(0, BLOCK_D)[None, :] * stride_q_d,
            mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & 
                 (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D),
            other=0.0
        )
        
        # Accumulate QK contributions from all N blocks
        for n_block_idx in range(num_n_blocks):
            n_block = tl.load(layout_csr_col_ptr + start_row + n_block_idx)
            
            # Calculate K offset
            k_seq_offset = n_block * BLOCK_N
            k_offset = (batch_idx * stride_k_batch + 
                        kv_head_idx * stride_k_head + 
                        k_seq_offset * stride_k_seq + 
                        d_block * BLOCK_D * stride_k_d)
            
            # Load K block and compute QK contribution
            k = tl.load(
                k_ptr + k_offset + 
                tl.arange(0, BLOCK_N)[None, :] * stride_k_seq + 
                tl.arange(0, BLOCK_D)[:, None] * stride_k_d,
                mask=(tl.arange(0, BLOCK_N)[None, :] < BLOCK_N) & 
                     (tl.arange(0, BLOCK_D)[:, None] < BLOCK_D),
                other=0.0
            )
            qk += tl.dot(q, k)
    
    # Apply softmax scaling
    qk *= softmax_scale
    
    # Compute attention weights with stable softmax
    max_qk = tl.max(qk, axis=1)
    exp_qk = tl.exp(qk - max_qk[:, None])
    sum_exp_qk = tl.sum(exp_qk, axis=1)
    softmax_qk = exp_qk / sum_exp_qk[:, None]
    
    # Calculate output offset
    out_seq_offset = m_block * BLOCK_M
    out_offset = (batch_idx * stride_out_batch + 
                  head_idx * stride_out_head + 
                  out_seq_offset * stride_out_seq)
    
    # Compute output contributions
    for d_block in range(NUM_D_BLOCKS):
        out_d = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        
        for n_block_idx in range(num_n_blocks):
            n_block = tl.load(layout_csr_col_ptr + start_row + n_block_idx)
            
            # Calculate V offset
            v_seq_offset = n_block * BLOCK_N
            v_offset = (batch_idx * stride_v_batch + 
                       kv_head_idx * stride_v_head + 
                       v_seq_offset * stride_v_seq + 
                       d_block * BLOCK_D * stride_v_d)
            
            # Load V block and accumulate output
            v = tl.load(
                v_ptr + v_offset + 
                tl.arange(0, BLOCK_N)[:, None] * stride_v_seq + 
                tl.arange(0, BLOCK_D)[None, :] * stride_v_d,
                mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & 
                     (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D),
                other=0.0
            )
            out_d += tl.dot(softmax_qk, v)
        
        # Store output block
        tl.store(
            out_ptr + out_offset + 
            d_block * BLOCK_D * stride_out_d + 
            tl.arange(0, BLOCK_M)[:, None] * stride_out_seq + 
            tl.arange(0, BLOCK_D)[None, :] * stride_out_d,
            out_d.to(out_ptr.dtype.element_ty),
            mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & 
                 (tl.arange(0, BLOCK_D)[None, :] < BLOCK_D)
        )

def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout_csr_row_indices: torch.Tensor,
    layout_csr_col_indices: torch.Tensor,
    softmax_scale: float,
    num_heads: int,
    num_kv_heads: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_D: int = 64,
):
    # Validate inputs
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert layout_csr_row_indices.is_cuda and layout_csr_col_indices.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert num_heads % num_kv_heads == 0
    
    batch, _, seq_len, d_model = q.shape
    device = q.device
    
    # Calculate grid dimensions
    num_q_blocks = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid = (num_q_blocks, batch * num_heads)
    
    # Prepare output tensor
    out = torch.empty_like(q)
    
    # Calculate number of D blocks
    NUM_D_BLOCKS = (d_model + BLOCK_D - 1) // BLOCK_D
    
    # Launch kernel
    block_sparse_attention_kernel[grid](
        q, k, v,
        layout_csr_row_indices, layout_csr_col_indices,
        out,
        # Strides for Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides for K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides for output
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        softmax_scale,
        num_heads,
        num_kv_heads,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=NUM_D_BLOCKS,
    )
    return out
