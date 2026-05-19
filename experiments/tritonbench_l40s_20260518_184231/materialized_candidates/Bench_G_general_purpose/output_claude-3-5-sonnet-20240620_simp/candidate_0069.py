import torch
import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr,
    # Sparse pattern metadata
    sparse_mask_ptr, block_tables_ptr,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Other parameters
    scale, causal,
    # Strides for the tensors
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    BLOCK_SIZE: tl.constexpr):
    
    # Compute indices for the current thread block
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    block_id = pid // (num_heads * num_blocks)
    head_id = (pid % (num_heads * num_blocks)) // num_blocks
    seq_block_id = pid % num_blocks

    # Compute start indices for this block
    start_seq_idx = seq_block_id * BLOCK_SIZE
    
    # Load block-specific metadata
    block_table = tl.load(block_tables_ptr + head_id * num_blocks + seq_block_id)
    
    # Create offsets for loading Q block
    offs_q = start_seq_idx + tl.arange(0, BLOCK_SIZE)
    mask_q = offs_q < seq_len
    
    # Initialize accumulator for output
    acc = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    
    # Load Q block
    q_block_ptr = q_ptr + block_id * stride_qb + head_id * stride_qh + start_seq_idx * stride_qs
    q = tl.load(q_block_ptr + offs_q[:, None] * stride_qs, mask=mask_q[:, None], other=0.0)
    
    # Process K,V blocks according to sparse pattern
    for k_block_idx in range(0, num_blocks):
        # Check if this block pair is in the sparse pattern
        if tl.load(sparse_mask_ptr + head_id * num_blocks * num_blocks + 
                  seq_block_id * num_blocks + k_block_idx):
            
            # Load K,V blocks
            k_start = k_block_idx * BLOCK_SIZE
            offs_k = k_start + tl.arange(0, BLOCK_SIZE)
            mask_k = offs_k < seq_len
            
            k_block_ptr = k_ptr + block_id * stride_kb + head_id * stride_kh + k_start * stride_ks
            v_block_ptr = v_ptr + block_id * stride_vb + head_id * stride_vh + k_start * stride_vs
            
            k = tl.load(k_block_ptr + offs_k[:, None] * stride_ks, mask=mask_k[:, None], other=0.0)
            v = tl.load(v_block_ptr + offs_k[:, None] * stride_vs, mask=mask_k[:, None], other=0.0)
            
            # Compute attention scores
            scores = tl.dot(q, k.transpose())
            scores = scores * scale
            
            # Apply causal mask if needed
            if causal:
                causal_mask = offs_q[:, None] >= offs_k[None, :]
                scores = tl.where(causal_mask, scores, float("-inf"))
            
            # Softmax
            scores = tl.softmax(scores)
            
            # Compute attention output
            output = tl.dot(scores, v)
            acc += output
    
    # Write output
    out_block_ptr = out_ptr + block_id * stride_ob + head_id * stride_oh + start_seq_idx * stride_os
    tl.store(out_block_ptr + offs_q[:, None] * stride_os, acc, mask=mask_q[:, None])

def _triton_mixed_sparse_attention(q, k, v, sparse_mask, block_tables, scale, causal=False):
    """
    Wrapper function for mixed sparse attention Triton kernel.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sparse_mask: Boolean tensor indicating which blocks should attend to each other
        block_tables: Tensor containing block-specific metadata
        scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
    
    Returns:
        output: Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Initialize output tensor
    output = torch.empty_like(q)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 64
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(seq_len, BLOCK_SIZE)
    
    # Launch kernel
    grid = (batch_size * num_heads * num_blocks,)
    
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, output,
        sparse_mask, block_tables,
        batch_size, num_heads, seq_len, head_dim,
        scale, causal,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
