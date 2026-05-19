import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, out,
    layout_csr_row_indices, layout_csr_col_indices,
    softmax_scale,
    batch_size, num_heads, num_kv_heads, num_q_blocks, num_k_blocks,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr
):
    # Get the block indices
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    
    # Compute the block indices in the query and key tensors
    q_block_idx = bid
    k_block_idx = layout_csr_col_indices[layout_csr_row_indices[q_block_idx]]
    
    # Compute the block indices in the value tensor
    v_block_idx = k_block_idx
    
    # Compute the block indices in the output tensor
    out_block_idx = q_block_idx
    
    # Compute the offsets for the query, key, and value blocks
    q_offset = (bid * BLOCK_M * BLOCK_D) + (hid * BLOCK_D)
    k_offset = (k_block_idx * BLOCK_N * BLOCK_D) + (hid * BLOCK_D)
    v_offset = (v_block_idx * BLOCK_N * BLOCK_D) + (hid * BLOCK_D)
    out_offset = (out_block_idx * BLOCK_M * BLOCK_D) + (hid * BLOCK_D)
    
    # Load the query, key, and value blocks
    q = tl.load(Q + q_offset, mask=q_offset < Q.size, other=0.0)
    k = tl.load(K + k_offset, mask=k_offset < K.size, other=0.0)
    v = tl.load(V + v_offset, mask=v_offset < V.size, other=0.0)
    
    # Compute the dot product between the query and key blocks
    qk = tl.dot(q, k, trans_b=True)
    
    # Scale the dot product
    qk *= softmax_scale
    
    # Compute the softmax
    qk = tl.softmax(qk)
    
    # Compute the attention-weighted values
    out_block = tl.dot(qk, v)
    
    # Store the result in the output tensor
    tl.store(out + out_offset, out_block, mask=out_offset < out.size)

import torch
import triton
import triton.language as tl

def block_sparse_attention(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, softmax_scale, num_heads, num_kv_heads, num_q_blocks, num_k_blocks, BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS):
    # Convert inputs to Triton tensors
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    layout_csr_row_indices = layout_csr_row_indices.contiguous()
    layout_csr_col_indices = layout_csr_col_indices.contiguous()
    
    # Get the batch size
    batch_size = Q.shape[0]
    
    # Allocate the output tensor
    out = torch.empty_like(Q)
    
    # Define the grid and block dimensions
    grid = (num_q_blocks, batch_size * num_heads)
    block = (BLOCK_M, BLOCK_N, BLOCK_D)
    
    # Launch the kernel
    block_sparse_attention_kernel[grid, block](
        Q, K, V, out,
        layout_csr_row_indices, layout_csr_col_indices,
        softmax_scale,
        batch_size, num_heads, num_kv_heads, num_q_blocks, num_k_blocks,
        BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS
    )
    
    return out
