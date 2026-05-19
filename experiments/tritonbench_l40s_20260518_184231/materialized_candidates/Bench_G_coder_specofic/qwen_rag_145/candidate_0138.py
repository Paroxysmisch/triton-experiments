import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    out,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    # Get the block index
    block_idx = tl.program_id(0)
    block_start = block_idx * BLOCK_M
    # Get the query, key, and value block indices
    q_block_indices = layout_csr_row_indices + block_start
    k_block_indices = layout_csr_col_indices
    v_block_indices = layout_csr_col_indices
    # Load the query, key, and value blocks
    q_blocks = tl.load(Q + q_block_indices)
    k_blocks = tl.load(K + k_block_indices)
    v_blocks = tl.load(V + v_block_indices)
    # Compute the dot product of the query and key blocks
    qk = tl.dot(q_blocks, k_blocks)
    # Scale the dot product by the softmax scale
    qk *= softmax_scale
    # Compute the softmax of the scaled dot product
    sm = tl.softmax(qk)
    # Compute the final output blocks by scaling the value blocks by the softmax
    out_blocks = tl.dot(sm, v_blocks)
    # Store the output blocks
    tl.store(out + block_start, out_blocks)

def block_sparse_attention_wrapper(
    Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out, softmax_scale
):
    # Compute the number of blocks
    num_blocks = (layout_csr_row_indices[-1] + 1) // BLOCK_M
    # Compute the grid size
    grid = (num_blocks,)
    # Launch the kernel
    block_sparse_attention_kernel[grid](
        Q,
        K,
        V,
        layout_csr_row_indices,
        layout_csr_col_indices,
        out,
        softmax_scale,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        NUM_D_BLOCKS,
    )
