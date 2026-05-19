import torch
import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, Out,
    block_count, block_offset, column_count, column_index,
    seqlens, sm_scale, qk_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    L, D,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    MAX_COLUMNS_PER_BLOCK: tl.constexpr,
):
    # Get program indices
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    block_q = tl.program_id(2)
    
    # Compute sequence length for current batch
    seq_len = tl.load(seqlens + b_idx)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if block_q >= num_blocks:
        return
    
    # Compute Q block start and offsets
    q_start = block_q * BLOCK_SIZE
    q_offsets = q_start + tl.arange(0, BLOCK_SIZE)
    q_mask = q_offsets < seq_len
    
    # Load Q block, shape (BLOCK_SIZE, D)
    q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + q_offsets[:, None] * stride_qs + tl.arange(0, D)[None, :] * stride_qd
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)
    q = q * qk_scale  # Scale Q
    
    # Load block_count for current Q block
    bc_ptr = block_count + b_idx * block_count.stride(0) + h_idx * block_count.stride(1) + block_q * block_count.stride(2)
    bc = tl.load(bc_ptr)
    bc = tl.minimum(bc, MAX_BLOCKS_PER_SEQ)  # Clamp to max allowed
    
    # Initialize online softmax variables
    max_scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    sum_exps = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    weighted_values = tl.zeros([BLOCK_SIZE, D], dtype=tl.float32)
    
    # Iterate over each possible block in the attended blocks
    for kb in range(MAX_BLOCKS_PER_SEQ):
        if kb >= bc:
            break
        
        # Load K block index
        block_k = tl.load(block_offset + b_idx * block_offset.stride(0) + h_idx * block_offset.stride(1) + block_q * block_offset.stride(2) + kb * block_offset.stride(3))
        if block_k >= num_blocks:  # Skip invalid blocks
            continue
        
        k_start = block_k * BLOCK_SIZE
        k_end = tl.minimum(k_start + BLOCK_SIZE, seq_len)
        if k_start >= seq_len:  # Skip empty K blocks
            continue
        
        # Load column_count for this K block
        cc_ptr = column_count + b_idx * column_count.stride(0) + h_idx * column_count.stride(1) + block_q * column_count.stride(2) + kb * column_count.stride(3)
        cc = tl.load(cc_ptr)
        cc = tl.minimum(cc, MAX_COLUMNS_PER_BLOCK)
        
        # Iterate over columns in the current K block
        for col in range(MAX_COLUMNS_PER_BLOCK):
            if col >= cc:
                break
            
            # Load column index within the K block
            col_idx = tl.load(column_index + b_idx * column_index.stride(0) + h_idx * column_index.stride(1) + block_q * column_index.stride(2) + kb * column_index.stride(3) + col * column_index.stride(4))
            k_pos = k_start + col_idx
            if k_pos >= seq_len:  # Skip invalid positions
                continue
            
            # Load K and V vectors
            k_ptr = K + b_idx * stride_kb + h_idx * stride_kh + k_pos * stride_ks + tl.arange(0, D) * stride_kd
            k = tl.load(k_ptr)
            v_ptr = V + b_idx * stride_vb + h_idx * stride_vh + k_pos * stride_vs + tl.arange(0, D) * stride_vd
            v = tl.load(v_ptr)
            
            # Compute scores: Q * K^T, scaled by sm_scale
            scores = tl.sum(q * k, axis=1) * sm_scale
            
            # Apply causal mask
            q_positions = q_offsets
            mask = q_positions >= k_pos
            scores = tl.where(mask, scores, float('-inf'))
            
            # Online softmax update
            current_max = tl.maximum(max_scores, scores)
            exp_old = tl.exp(max_scores - current_max)
            exp_new = tl.exp(scores - current_max)
            
            sum_exp_new = sum_exps * exp_old + exp_new
            weighted_new = weighted_values * exp_old[:, None] + exp_new[:, None] * v
            
            # Update accumulators
            sum_exps = sum_exp_new
            weighted_values = weighted_new
            max_scores = current_max
    
    # Normalize the weighted values by sum_exps to get final output
    out = weighted_values / sum_exps[:, None]
    
    # Write back to output tensor
    out_ptr = Out + b_idx * stride_ob + h_idx * stride_oh + q_offsets[:, None] * stride_os + tl.arange(0, D)[None, :] * stride_od
    tl.store(out_ptr, out.to(out_ptr.dtype.element_ty), mask=q_mask[:, None])

def _triton_mixed_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    seqlens: torch.Tensor,
    sm_scale: float,
    qk_scale: float,
    BLOCK_SIZE: int = 64,
    MAX_BLOCKS_PER_SEQ: int = 8,
    MAX_COLUMNS_PER_BLOCK: int = 32,
):
    # Check tensor shapes
    B, H, L, D = Q.shape
    assert K.shape == (B, H, L, D)
    assert V.shape == (B, H, L, D)
    assert block_count.shape == (B, H, (L + BLOCK_SIZE - 1) // BLOCK_SIZE)
    assert block_offset.shape == (B, H, block_count.shape[2], MAX_BLOCKS_PER_SEQ)
    assert column_count.shape == (B, H, block_count.shape[2], MAX_BLOCKS_PER_SEQ)
    assert column_index.shape == (B, H, block_count.shape[2], MAX_BLOCKS_PER_SEQ, MAX_COLUMNS_PER_BLOCK)
    assert seqlens.shape == (B,)
    
    # Allocate output tensor
    Out = torch.empty_like(Q)
    
    # Grid configuration
    num_blocks_q = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B, H, num_blocks_q)
    
    # Launch kernel
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        Q, K, V, Out,
        block_count, block_offset, column_count, column_index,
        seqlens, sm_scale, qk_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        L, D,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        MAX_COLUMNS_PER_BLOCK=MAX_COLUMNS_PER_BLOCK,
    )
    
    return Out
