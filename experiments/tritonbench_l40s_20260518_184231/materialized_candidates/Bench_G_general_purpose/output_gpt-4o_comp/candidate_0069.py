import triton
import triton.language as tl

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, Out,
    block_count, block_offset, column_count, column_index,
    seqlens, sm_scale, qk_scale,
    stride_q, stride_k, stride_v, stride_out,
    stride_block_count, stride_block_offset,
    stride_column_count, stride_column_index,
    BLOCK_SIZE: tl.constexpr, MAX_BLOCK_COUNT: tl.constexpr, MAX_COLUMN_COUNT: tl.constexpr
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Load the block indices for the current program
    block_idx = tl.load(block_offset + pid * stride_block_offset)
    num_blocks = tl.load(block_count + pid * stride_block_count)
    
    # Initialize accumulators for the output
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Iterate over blocks
    for block in range(num_blocks):
        # Load block indices
        col_idx = tl.load(column_index + block * stride_column_index)
        num_cols = tl.load(column_count + block * stride_column_count)
        
        # Iterate over columns
        for col in range(num_cols):
            # Calculate the offsets for Q, K, V
            q_offset = (block_idx + block) * BLOCK_SIZE
            k_offset = col_idx * BLOCK_SIZE
            v_offset = col_idx * BLOCK_SIZE
            
            # Load Q, K, V blocks
            q = tl.load(Q + q_offset * stride_q + tl.arange(0, BLOCK_SIZE))
            k = tl.load(K + k_offset * stride_k + tl.arange(0, BLOCK_SIZE))
            v = tl.load(V + v_offset * stride_v + tl.arange(0, BLOCK_SIZE))
            
            # Scale Q and compute attention scores
            q_scaled = q * qk_scale
            attn_scores = tl.dot(q_scaled, k.T) * sm_scale
            
            # Apply causal mask
            mask = tl.arange(0, BLOCK_SIZE)[:, None] >= tl.arange(0, BLOCK_SIZE)[None, :]
            attn_scores = tl.where(mask, attn_scores, float('-inf'))
            
            # Softmax and accumulate
            attn_weights = tl.softmax(attn_scores)
            acc += tl.dot(attn_weights, v)
    
    # Write the result to the output tensor
    tl.store(Out + q_offset * stride_out + tl.arange(0, BLOCK_SIZE), acc)

def _triton_mixed_sparse_attention(Q, K, V, block_count, block_offset, column_count, column_index, seqlens, sm_scale, qk_scale, BLOCK_SIZE=128):
    # Define the shape and strides of the input tensors
    stride_q = Q.stride(0)
    stride_k = K.stride(0)
    stride_v = V.stride(0)
    stride_out = Q.stride(0)  # Assuming output has the same stride as Q
    
    stride_block_count = block_count.stride(0)
    stride_block_offset = block_offset.stride(0)
    stride_column_count = column_count.stride(0)
    stride_column_index = column_index.stride(0)
    
    # Configure the grid
    grid = (len(block_count),)
    
    # Launch the kernel
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        Q, K, V, Out,
        block_count, block_offset, column_count, column_index,
        seqlens, sm_scale, qk_scale,
        stride_q, stride_k, stride_v, stride_out,
        stride_block_count, stride_block_offset,
        stride_column_count, stride_column_index,
        BLOCK_SIZE=BLOCK_SIZE, MAX_BLOCK_COUNT=128, MAX_COLUMN_COUNT=128
    )
