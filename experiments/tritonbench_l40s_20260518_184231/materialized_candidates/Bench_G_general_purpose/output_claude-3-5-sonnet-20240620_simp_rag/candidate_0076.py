@triton.jit
def _sampled_addmm_kernel(
    alpha, beta, IS_BETA_ZERO: tl.constexpr,
    BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr,
    k, TILE_K: tl.constexpr,
    values_ptr, values_batch_stride, values_nnz_stride,
    values_row_block_stride, values_col_block_stride,
    crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride,
    col_indices_ptr, col_indices_batch_stride, col_indices_stride,
    mat1_ptr, mat1_batch_stride, mat1_tiled_row_stride,
    mat1_tiled_col_stride, mat1_row_block_stride, mat1_col_block_stride,
    mat2_ptr, mat2_batch_stride, mat2_tiled_row_stride,
    mat2_tiled_col_stride, mat2_row_block_stride, mat2_col_block_stride,
    acc_dtype: tl.constexpr, allow_tf32: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate batch and row indices
    batch_idx = pid // (k // TILE_K)
    k_idx = pid % (k // TILE_K)
    
    # Load row offsets
    row_start = tl.load(crow_indices_ptr + batch_idx * crow_indices_batch_stride + k_idx * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + batch_idx * crow_indices_batch_stride + (k_idx + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCKSIZE_ROW, BLOCKSIZE_COL], dtype=acc_dtype)
    
    # Loop over non-zero blocks
    for i in range(row_start, row_end):
        # Load column index
        col = tl.load(col_indices_ptr + batch_idx * col_indices_batch_stride + i * col_indices_stride)
        
        # Load values block
        values = tl.load(values_ptr + 
                        batch_idx * values_batch_stride +
                        i * values_nnz_stride +
                        tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
                        tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        
        # Load mat1 and mat2 blocks
        mat1_block = tl.load(mat1_ptr +
                            batch_idx * mat1_batch_stride +
                            k_idx * mat1_tiled_row_stride +
                            tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat1_row_block_stride +
                            tl.arange(0, BLOCKSIZE_COL)[None, :] * mat1_col_block_stride)
        
        mat2_block = tl.load(mat2_ptr +
                            batch_idx * mat2_batch_stride +
                            col * mat2_tiled_row_stride +
                            tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat2_row_block_stride +
                            tl.arange(0, BLOCKSIZE_COL)[None, :] * mat2_col_block_stride)
        
        # Compute block multiplication
        acc += alpha * tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32)
    
    # Apply beta scaling if needed
    if not IS_BETA_ZERO:
        acc = acc * beta + values
    
    # Store result
    tl.store(values_ptr +
             batch_idx * values_batch_stride +
             k_idx * values_nnz_stride +
             tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride,
             acc)
