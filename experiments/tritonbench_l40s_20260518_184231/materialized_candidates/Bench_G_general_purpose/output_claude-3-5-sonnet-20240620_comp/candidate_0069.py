import torch
import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr,
    # Block indices and counts
    block_count_ptr, block_offset_ptr,
    column_count_ptr, column_index_ptr,
    # Sequence lengths and scaling
    seqlens_ptr, sm_scale,
    # Matrix dimensions
    batch_size, num_heads, head_dim,
    max_block_count, max_column_count,
    block_size, BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Load sequence length for this batch
    seqlen = tl.load(seqlens_ptr + batch_id)
    
    # Compute offsets
    batch_offset = batch_id * num_heads * head_dim
    head_offset = head_id * head_dim
    base_offset = batch_offset + head_offset

    # Allocate shared memory for block processing
    block_k = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    block_v = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    block_scores = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Load block information
    block_count = tl.load(block_count_ptr + pid)
    block_offset_start = tl.load(block_offset_ptr + pid)

    # Process each block
    for block_idx in range(block_count):
        # Load column information
        col_count = tl.load(column_count_ptr + block_offset_start + block_idx)
        col_offset = tl.load(block_offset_ptr + block_offset_start + block_idx)

        # Process each column in the block
        for col_idx in range(0, col_count, BLOCK_SIZE):
            # Load column indices
            col_indices = tl.load(column_index_ptr + col_offset + col_idx + tl.arange(0, BLOCK_SIZE))
            
            # Load K and V blocks
            k_block_ptr = k_ptr + base_offset + col_indices * head_dim
            v_block_ptr = v_ptr + base_offset + col_indices * head_dim
            
            # Load K and V data
            for i in range(0, head_dim, BLOCK_SIZE):
                cols = tl.arange(0, BLOCK_SIZE)
                mask = cols < head_dim - i
                block_k[:, i:i+BLOCK_SIZE] = tl.load(k_block_ptr + i, mask=mask)
                block_v[:, i:i+BLOCK_SIZE] = tl.load(v_block_ptr + i, mask=mask)

            # Compute attention scores
            q_idx = pid * head_dim
            q_block = tl.load(q_ptr + q_idx + tl.arange(0, head_dim))
            
            # Scaled dot product
            scores = tl.sum(q_block[:, None] * block_k, axis=1) * sm_scale
            
            # Apply causal masking if needed
            causal_mask = col_indices[:, None] <= tl.arange(0, seqlen)[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))
            
            # Softmax
            scores = tl.softmax(scores)
            
            # Compute weighted sum
            out = tl.sum(scores[:, None] * block_v, axis=0)
            
            # Accumulate to output
            out_idx = pid * head_dim
            tl.store(out_ptr + out_idx + tl.arange(0, head_dim), out)

def _triton_mixed_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    seqlens: torch.Tensor,
    sm_scale: float,
    BLOCK_SIZE: int = 32
):
    batch_size, num_heads, seq_len, head_dim = q.shape
    max_block_count = block_count.max().item()
    max_column_count = column_count.max().item()
    
    # Prepare output tensor
    output = torch.empty_like(q)
    
    # Launch kernel
    grid = (batch_size * num_heads,)
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, output,
        block_count, block_offset,
        column_count, column_index,
        seqlens, sm_scale,
        batch_size, num_heads, head_dim,
        max_block_count, max_column_count,
        BLOCK_SIZE, BLOCK_SIZE
    )
    
    return output
