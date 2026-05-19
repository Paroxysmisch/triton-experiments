import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

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
    
    # Load row indices for current block
    batch_idx = pid // (k * TILE_K)
    row_idx = (pid % (k * TILE_K)) // TILE_K
    tile_idx = pid % TILE_K
    
    # Compute pointers for current batch
    values_batch_ptr = values_ptr + batch_idx * values_batch_stride
    crow_batch_ptr = crow_indices_ptr + batch_idx * crow_indices_batch_stride
    col_batch_ptr = col_indices_ptr + batch_idx * col_indices_batch_stride
    
    # Load row start and end indices
    row_start = tl.load(crow_batch_ptr + row_idx * crow_indices_stride)
    row_end = tl.load(crow_batch_ptr + (row_idx + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
    
    # Loop over non-zero blocks in current row
    for nnz_idx in range(row_start, row_end):
        # Load column index
        col_idx = tl.load(col_batch_ptr + nnz_idx * col_indices_stride)
        
        # Load values block
        values = tl.load(values_batch_ptr + 
                        nnz_idx * values_nnz_stride +
                        tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
                        tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        
        # Load mat1 and mat2 blocks
        mat1_block = tl.load(mat1_ptr + 
                           batch_idx * mat1_batch_stride +
                           row_idx * mat1_tiled_row_stride +
                           col_idx * mat1_tiled_col_stride +
                           tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat1_row_block_stride +
                           tl.arange(0, BLOCKSIZE_COL)[None, :] * mat1_col_block_stride)
        
        mat2_block = tl.load(mat2_ptr +
                           batch_idx * mat2_batch_stride +
                           col_idx * mat2_tiled_row_stride +
                           tile_idx * mat2_tiled_col_stride +
                           tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat2_row_block_stride +
                           tl.arange(0, BLOCKSIZE_COL)[None, :] * mat2_col_block_stride)
        
        # Compute block multiplication and accumulate
        if allow_tf32:
            acc += tl.dot(mat1_block, mat2_block, allow_tf32=True)
        else:
            acc += tl.dot(mat1_block, mat2_block)
    
    # Apply alpha scaling
    acc = acc * alpha
    
    # Apply beta scaling if needed
    if not IS_BETA_ZERO:
        acc = acc * beta + values
    
    # Store result
    tl.store(values_batch_ptr + 
             row_idx * values_row_block_stride +
             tile_idx * values_col_block_stride +
             tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride,
             acc)

def _run_sampled_addmm_kernel(
    alpha, beta, is_beta_zero,
    blocksize, k, tile_k,
    values, crow_indices, col_indices,
    mat1, mat2,
    max_grid
):
    # Configure kernel parameters
    BLOCKSIZE_ROW, BLOCKSIZE_COL = blocksize
    
    # Launch kernel
    grid = (k * tile_k,)
    if max_grid is not None:
        grid = tuple(min(g, m) if m is not None else g 
                    for g, m in zip(grid, max_grid))
    
    _sampled_addmm_kernel[grid](
        alpha=alpha, beta=beta,
        IS_BETA_ZERO=is_beta_zero,
        BLOCKSIZE_ROW=BLOCKSIZE_ROW,
        BLOCKSIZE_COL=BLOCKSIZE_COL,
        k=k, TILE_K=tile_k,
        values_ptr=values.data_ptr(),
        values_batch_stride=values.stride(0),
        values_nnz_stride=values.stride(1),
        values_row_block_stride=values.stride(2),
        values_col_block_stride=values.stride(3),
        crow_indices_ptr=crow_indices.data_ptr(),
        crow_indices_batch_stride=crow_indices.stride(0),
        crow_indices_stride=crow_indices.stride(1),
        col_indices_ptr=col_indices.data_ptr(),
        col_indices_batch_stride=col_indices.stride(0),
        col_indices_stride=col_indices.stride(1),
        mat1_ptr=mat1.data_ptr(),
        mat1_batch_stride=mat1.stride(0),
        mat1_tiled_row_stride=mat1.stride(1),
        mat1_tiled_col_stride=mat1.stride(2),
        mat1_row_block_stride=mat1.stride(3),
        mat1_col_block_stride=mat1.stride(4),
        mat2_ptr=mat2.data_ptr(),
        mat2_batch_stride=mat2.stride(0),
        mat2_tiled_row_stride=mat2.stride(1),
        mat2_tiled_col_stride=mat2.stride(2),
        mat2_row_block_stride=mat2.stride(3),
        mat2_col_block_stride=mat2.stride(4),
        acc_dtype=tl.float32,
        allow_tf32=True,
    )
