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
    
    # Calculate batch and row indices
    batch_idx = pid // (k // BLOCKSIZE_ROW)
    row_block_idx = pid % (k // BLOCKSIZE_ROW)
    
    # Load row offsets
    row_start = tl.load(crow_indices_ptr + batch_idx * crow_indices_batch_stride + 
                       row_block_idx * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + batch_idx * crow_indices_batch_stride + 
                      (row_block_idx + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
    
    # Loop over non-zero blocks in the row
    for nnz_idx in range(row_start, row_end):
        # Load column index
        col_idx = tl.load(col_indices_ptr + batch_idx * col_indices_batch_stride + 
                         nnz_idx * col_indices_stride)
        
        # Load values block
        values = tl.load(values_ptr + batch_idx * values_batch_stride + 
                        nnz_idx * values_nnz_stride +
                        tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
                        tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        
        # Load mat2 block
        mat2_block = tl.load(mat2_ptr + batch_idx * mat2_batch_stride +
                            col_idx * mat2_tiled_col_stride +
                            tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat2_row_block_stride +
                            tl.arange(0, BLOCKSIZE_COL)[None, :] * mat2_col_block_stride)
        
        # Accumulate product
        acc += values * mat2_block
    
    # Scale by alpha
    acc = acc * alpha
    
    # Add beta * mat1 if needed
    if not IS_BETA_ZERO:
        mat1_block = tl.load(mat1_ptr + batch_idx * mat1_batch_stride +
                            row_block_idx * mat1_tiled_row_stride +
                            tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat1_row_block_stride +
                            tl.arange(0, BLOCKSIZE_COL)[None, :] * mat1_col_block_stride)
        acc += beta * mat1_block
    
    # Store result back to values
    tl.store(values_ptr + batch_idx * values_batch_stride +
             row_block_idx * values_nnz_stride +
             tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride,
             acc)
