import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

# Sampled AddMM Kernel
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

    # Compute the block ID and the starting index in the crow_indices array
    block_id = pid // (BLOCKSIZE_ROW * BLOCKSIZE_COL)
    block_offset = pid % (BLOCKSIZE_ROW * BLOCKSIZE_COL)
    row_id = block_offset // BLOCKSIZE_COL
    col_id = block_offset % BLOCKSIZE_COL

    # Load the values and indices
    values = tl.load(values_ptr + batch_id * values_batch_stride + block_id * values_nnz_stride)
    crow_indices = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + row_id * crow_indices_stride)
    col_indices = tl.load(col_indices_ptr + batch_id * col_indices_batch_stride + col_id * col_indices_stride)

    # Initialize the output
    output = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    # Perform the sampled addmm operation
    for i in range(0, k, TILE_K):
        mat1 = tl.load(mat1_ptr + batch_id * mat1_batch_stride + row_id * mat1_tiled_row_stride + i * mat1_tiled_col_stride)
        mat2 = tl.load(mat2_ptr + batch_id * mat2_batch_stride + col_id * mat2_tiled_row_stride + i * mat2_tiled_col_stride)
        output += alpha * tl.dot(mat1, mat2)

    if not IS_BETA_ZERO:
        output += beta * values

    # Store the result
    tl.store(values_ptr + batch_id * values_batch_stride + block_id * values_nnz_stride, output)

# BSR Strided Dense Row Space Kernel
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

    # Compute the block ID and the starting index in the crow_indices array
    block_id = pid // (BLOCKSIZE_ROW * BLOCKSIZE_COL)
    block_offset = pid % (BLOCKSIZE_ROW * BLOCKSIZE_COL)
    row_id = block_offset // BLOCKSIZE_COL
    col_id = block_offset % BLOCKSIZE_COL

    # Load the values and indices
    values = tl.load(values_ptr + batch_id * values_batch_stride + block_id * values_nnz_stride)
    crow_indices = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + row_id * crow_indices_stride)
    col_indices = tl.load(col_indices_ptr + batch_id * col_indices_batch_stride + col_id * col_indices_stride)

    # Initialize the output
    output = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    # Perform the dense rowspace operation
    for i in range(0, BLOCKSIZE_ROW, GROUP_SIZE_ROW):
        dense = tl.load(dense_ptr + batch_id * dense_batch_stride + row_id * dense_tiled_row_stride + i * dense_tiled_col_stride)
        output += tl.dot(values, dense)

    # Store the result
    tl.store(output_ptr + batch_id * output_batch_stride + block_id * output_nnz_stride, output)

# BSR Softmax Kernel
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

    # Compute the block ID and the starting index in the crow_indices array
    block_id = pid // (row_block * col_block)
    block_offset = pid % (row_block * col_block)
    row_id = block_offset // col_block
    col_id = block_offset % col_block

    # Load the values and indices
    values = tl.load(values_ptr + batch_id * values_batch_stride + block_id * values_nnz_col_block_stride)
    crow_indices = tl.load(crow_indices_ptr + batch_id * crow_indices_batch_stride + row_id * crow_indices_stride)

    # Initialize the output
    output = tl.zeros((row_block, col_block), dtype=tl.float32)

    # Perform the softmax operation
    for i in range(0, MAX_ROW_NNZ, TILE):
        values_tile = tl.load(values + i * values_row_block_stride)
        max_val = tl.max(values_tile, axis=0)
        exp_val = tl.exp(values_tile - max_val)
        sum_exp = tl.sum(exp_val, axis=0)
        softmax_val = exp_val / sum_exp
        output += softmax_val

    # Store the result
    tl.store(values_ptr + batch_id * values_batch_stride + block_id * values_nnz_col_block_stride, output)
