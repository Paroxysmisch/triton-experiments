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
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    block_id = pid % num_blocks

    row = block_id // (BLOCKSIZE_COL // TILE_K)
    col = block_id % (BLOCKSIZE_COL // TILE_K)

    values_offset = (row * values_row_block_stride + col * values_col_block_stride) * values_nnz_stride
    mat1_offset = (row * mat1_row_block_stride + col * mat1_col_block_stride) * mat1_tiled_row_stride
    mat2_offset = (row * mat2_row_block_stride + col * mat2_col_block_stride) * mat2_tiled_row_stride

    values = tl.load(values_ptr + values_offset)
    mat1 = tl.load(mat1_ptr + mat1_offset)
    mat2 = tl.load(mat2_ptr + mat2_offset)

    if IS_BETA_ZERO:
        output = alpha * tl.dot(mat1, mat2)
    else:
        output = beta * values + alpha * tl.dot(mat1, mat2)

    tl.store(values_ptr + values_offset, output)

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
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    block_id = pid % num_blocks

    row = block_id // (BLOCKSIZE_COL // GROUP_SIZE_ROW)
    col = block_id % (BLOCKSIZE_COL // GROUP_SIZE_ROW)

    values_offset = (row * values_row_block_stride + col * values_col_block_stride) * values_nnz_stride
    crow_offset = row * crow_indices_stride
    col_offset = col * col_indices_stride
    dense_offset = (row * dense_row_block_stride + col * dense_col_block_stride) * dense_tiled_row_stride
    output_offset = (row * output_row_block_stride + col * output_col_block_stride) * output_tiled_row_stride

    values = tl.load(values_ptr + values_offset)
    crow_indices = tl.load(crow_indices_ptr + crow_offset)
    col_indices = tl.load(col_indices_ptr + col_offset)
    dense = tl.load(dense_ptr + dense_offset)

    output = tl.dot(values, dense)
    tl.store(output_ptr + output_offset, output)

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
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    block_id = pid % num_blocks

    row = block_id // (col_block // TILE)
    col = block_id % (col_block // TILE)

    values_offset = (row * values_row_block_stride + col * values_nnz_col_block_stride) * values_batch_stride
    crow_offset = row * crow_indices_stride

    values = tl.load(values_ptr + values_offset)
    crow_indices = tl.load(crow_indices_ptr + crow_offset)

    max_val = tl.max(values, axis=1)
    values = values - max_val[:, None]
    exp_values = tl.exp(values)
    sum_exp = tl.sum(exp_values, axis=1)
    softmax_values = exp_values / sum_exp[:, None]

    tl.store(values_ptr + values_offset, softmax_values)
