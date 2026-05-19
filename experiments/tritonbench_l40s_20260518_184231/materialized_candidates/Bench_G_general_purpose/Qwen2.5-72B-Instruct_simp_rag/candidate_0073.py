import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _sampled_addmm_kernel(
    alpha,
    beta,
    IS_BETA_ZERO: tl.constexpr,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    k,
    TILE_K: tl.constexpr,
    values_ptr,
    values_batch_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    mat1_ptr,
    mat1_batch_stride,
    mat1_tiled_row_stride,
    mat1_tiled_col_stride,
    mat1_row_block_stride,
    mat1_col_block_stride,
    mat2_ptr,
    mat2_batch_stride,
    mat2_tiled_row_stride,
    mat2_tiled_col_stride,
    mat2_row_block_stride,
    mat2_col_block_stride,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    # Kernel implementation here
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    
    # Compute the row and column indices
    row_start = crow_indices_ptr[batch_id * crow_indices_batch_stride + pid * crow_indices_stride]
    row_end = crow_indices_ptr[batch_id * crow_indices_batch_stride + (pid + 1) * crow_indices_stride]
    nnz = row_end - row_start
    
    for n in range(0, nnz, BLOCKSIZE_ROW):
        for m in range(0, BLOCKSIZE_COL, BLOCKSIZE_COL):
            acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
            for p in range(0, k, TILE_K):
                a = tl.load(values_ptr + batch_id * values_batch_stride + (row_start + n) * values_nnz_stride + p * values_row_block_stride)
                b = tl.load(mat1_ptr + batch_id * mat1_batch_stride + (row_start + n) * mat1_tiled_row_stride + p * mat1_row_block_stride)
                c = tl.load(mat2_ptr + batch_id * mat2_batch_stride + p * mat2_tiled_row_stride + m * mat2_col_block_stride)
                acc += tl.dot(a, b, allow_tf32=allow_tf32) * alpha + c * beta
            tl.store(values_ptr + batch_id * values_batch_stride + (row_start + n) * values_nnz_stride + m * values_col_block_stride, acc)

@triton.jit
def _bsr_strided_dense_rowspace_kernel(
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    values_ptr,
    values_batch_stride,
    values_nnz_stride,
    values_row_block_stride,
    values_col_block_stride,
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    col_indices_ptr,
    col_indices_batch_stride,
    col_indices_stride,
    dense_ptr,
    dense_batch_stride,
    dense_tiled_row_stride,
    dense_tiled_col_stride,
    dense_row_block_stride,
    dense_col_block_stride,
    output_ptr,
    output_batch_stride,
    output_tiled_row_stride,
    output_tiled_col_stride,
    output_row_block_stride,
    output_col_block_stride,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    GROUP_SIZE_ROW: tl.constexpr,
):
    # Kernel implementation here
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    
    # Compute the row and column indices
    row_start = crow_indices_ptr[batch_id * crow_indices_batch_stride + pid * crow_indices_stride]
    row_end = crow_indices_ptr[batch_id * crow_indices_batch_stride + (pid + 1) * crow_indices_stride]
    nnz = row_end - row_start
    
    for n in range(0, nnz, BLOCKSIZE_ROW):
        for m in range(0, BLOCKSIZE_COL, BLOCKSIZE_COL):
            acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
            for p in range(0, BLOCKSIZE_COL, BLOCKSIZE_COL):
                a = tl.load(values_ptr + batch_id * values_batch_stride + (row_start + n) * values_nnz_stride + p * values_row_block_stride)
                b = tl.load(dense_ptr + batch_id * dense_batch_stride + (row_start + n) * dense_tiled_row_stride + p * dense_row_block_stride)
                acc += tl.dot(a, b, allow_tf32=allow_tf32)
            tl.store(output_ptr + batch_id * output_batch_stride + (row_start + n) * output_tiled_row_stride + m * output_col_block_stride, acc)

@triton.jit
def _bsr_softmax_kernel(
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    values_ptr,
    values_batch_stride,
    values_row_block_stride,
    values_nnz_col_block_stride,
    row_block, col_block,
    MAX_ROW_NNZ: tl.constexpr,
    TILE: tl.constexpr
):
    # Kernel implementation here
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    
    # Compute the row and column indices
    row_start = crow_indices_ptr[batch_id * crow_indices_batch_stride + pid * crow_indices_stride]
    row_end = crow_indices_ptr[batch_id * crow_indices_batch_stride + (pid + 1) * crow_indices_stride]
    nnz = row_end - row_start
    
    for n in range(0, nnz, TILE):
        values = tl.load(values_ptr + batch_id * values_batch_stride + (row_start + n) * values_row_block_stride)
        max_val = tl.max(values, axis=0)
        values = values - max_val
        exp_values = tl.exp(values)
        sum_exp = tl.sum(exp_values, axis=0)
        softmax_values = exp_values / sum_exp
        tl.store(values_ptr + batch_id * values_batch_stride + (row_start + n) * values_row_block_stride, softmax_values)
