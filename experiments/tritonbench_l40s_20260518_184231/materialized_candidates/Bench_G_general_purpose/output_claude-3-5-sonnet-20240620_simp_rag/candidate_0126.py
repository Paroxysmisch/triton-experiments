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
    # Get program ID and compute batch/row indices
    pid = tl.program_id(0)
    batch_id = pid // k
    row_id = pid % k

    # Load row offsets for current batch and row
    row_start = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + 
                       row_id * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + 
                      (row_id + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCKSIZE_ROW, BLOCKSIZE_COL], dtype=acc_dtype)
    
    # Loop over non-zero blocks in current row
    for nnz_idx in range(row_start, row_end):
        # Load column index
        col_idx = tl.load(col_indices_ptr + batch_id * col_indices_batch_stride + 
                         nnz_idx * col_indices_stride)
        
        # Load blocks from input matrices
        a_block_ptr = values_ptr + (batch_id * values_batch_stride + 
                                  nnz_idx * values_nnz_stride)
        b_block_ptr = mat2_ptr + (batch_id * mat2_batch_stride + 
                                 col_idx * mat2_tiled_col_stride)
        
        # Load blocks
        a = tl.load(a_block_ptr + tl.arange(0, BLOCKSIZE_ROW)[:, None] * 
                   values_row_block_stride + 
                   tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        b = tl.load(b_block_ptr + tl.arange(0, BLOCKSIZE_COL)[:, None] * 
                   mat2_row_block_stride + 
                   tl.arange(0, TILE_K)[None, :] * mat2_col_block_stride)
        
        # Accumulate product
        acc += tl.dot(a, b, allow_tf32=allow_tf32)
    
    # Scale accumulator by alpha
    acc = acc * alpha
    
    # Add beta * input if needed
    if not IS_BETA_ZERO:
        c_block_ptr = mat1_ptr + (batch_id * mat1_batch_stride + 
                                 row_id * mat1_tiled_row_stride)
        c = tl.load(c_block_ptr + tl.arange(0, BLOCKSIZE_ROW)[:, None] * 
                   mat1_row_block_stride + 
                   tl.arange(0, BLOCKSIZE_COL)[None, :] * mat1_col_block_stride)
        acc += beta * c
    
    # Store result
    output_ptr = values_ptr + (batch_id * values_batch_stride + 
                              row_id * values_nnz_stride)
    tl.store(output_ptr + tl.arange(0, BLOCKSIZE_ROW)[:, None] * 
             values_row_block_stride + 
             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride, acc)

# Wrapper function to launch the kernel
def _run_sampled_addmm_kernel(
    alpha, beta, is_beta_zero,
    blocksize, k, tile_k,
    values, crow_indices, col_indices,
    mat1, mat2,
    max_grid
):
    batch_size = values.size(0)
    grid = (batch_size * k,)
    
    # Launch kernel
    _sampled_addmm_kernel[grid](
        alpha, beta, is_beta_zero,
        blocksize[0], blocksize[1], k, tile_k,
        values.data_ptr(), values.stride(0), values.stride(1),
        values.stride(2), values.stride(3),
        crow_indices.data_ptr(), crow_indices.stride(0), crow_indices.stride(1),
        col_indices.data_ptr(), col_indices.stride(0), col_indices.stride(1),
        mat1.data_ptr(), mat1.stride(0), mat1.stride(1),
        mat1.stride(2), mat1.stride(3), mat1.stride(4),
        mat2.data_ptr(), mat2.stride(0), mat2.stride(1),
        mat2.stride(2), mat2.stride(3), mat2.stride(4),
        tl.float32, True
    )
