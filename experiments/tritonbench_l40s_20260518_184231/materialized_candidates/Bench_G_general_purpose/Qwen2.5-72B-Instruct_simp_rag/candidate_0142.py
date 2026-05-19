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

    # Compute the block indices
    block_row = block_id // (BLOCKSIZE_COL // TILE_K)
    block_col = block_id % (BLOCKSIZE_COL // TILE_K)

    # Compute the block offset
    block_offset = block_row * (BLOCKSIZE_COL // TILE_K) + block_col

    # Load the block indices
    crow_start = tl.load(crow_indices_ptr + block_row * crow_indices_stride)
    crow_end = tl.load(crow_indices_ptr + (block_row + 1) * crow_indices_stride)

    # Initialize the accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    for nnz in range(crow_start, crow_end):
        col = tl.load(col_indices_ptr + nnz * col_indices_stride)
        value = tl.load(values_ptr + nnz * values_nnz_stride)

        # Load the matrix blocks
        mat1_block = tl.load(mat1_ptr + nnz * mat1_tiled_row_stride + block_offset * mat1_tiled_col_stride)
        mat2_block = tl.load(mat2_ptr + col * mat2_tiled_row_stride + block_offset * mat2_tiled_col_stride)

        # Perform the matrix multiplication
        acc += value * tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32)

    # Add the bias term if beta is not zero
    if not IS_BETA_ZERO:
        acc += beta * tl.load(mat1_ptr + block_row * mat1_tiled_row_stride + block_col * mat1_tiled_col_stride)

    # Store the result
    tl.store(mat1_ptr + block_row * mat1_tiled_row_stride + block_col * mat1_tiled_col_stride, acc)

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

    # Compute the block indices
    block_row = block_id // (BLOCKSIZE_COL // GROUP_SIZE_ROW)
    block_col = block_id % (BLOCKSIZE_COL // GROUP_SIZE_ROW)

    # Compute the block offset
    block_offset = block_row * (BLOCKSIZE_COL // GROUP_SIZE_ROW) + block_col

    # Load the block indices
    crow_start = tl.load(crow_indices_ptr + block_row * crow_indices_stride)
    crow_end = tl.load(crow_indices_ptr + (block_row + 1) * crow_indices_stride)

    # Initialize the accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    for nnz in range(crow_start, crow_end):
        col = tl.load(col_indices_ptr + nnz * col_indices_stride)
        value = tl.load(values_ptr + nnz * values_nnz_stride)

        # Load the matrix blocks
        dense_block = tl.load(dense_ptr + col * dense_tiled_row_stride + block_offset * dense_tiled_col_stride)

        # Perform the matrix multiplication
        acc += value * dense_block

    # Store the result
    tl.store(output_ptr + block_row * output_tiled_row_stride + block_col * output_tiled_col_stride, acc)

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

    # Compute the block indices
    block_row = block_id // (col_block // TILE)
    block_col = block_id % (col_block // TILE)

    # Compute the block offset
    block_offset = block_row * (col_block // TILE) + block_col

    # Load the block indices
    crow_start = tl.load(crow_indices_ptr + block_row * crow_indices_stride)
    crow_end = tl.load(crow_indices_ptr + (block_row + 1) * crow_indices_stride)

    # Initialize the accumulator
    max_val = -float('inf')
    sum_exp = 0.0

    for nnz in range(crow_start, crow_end):
        value = tl.load(values_ptr + nnz * values_nnz_col_block_stride + block_offset * values_row_block_stride)

        # Compute the maximum value
        max_val = tl.max(max_val, value)

    for nnz in range(crow_start, crow_end):
        value = tl.load(values_ptr + nnz * values_nnz_col_block_stride + block_offset * values_row_block_stride)

        # Compute the exponent and sum
        exp_val = tl.exp(value - max_val)
        sum_exp += exp_val

    for nnz in range(crow_start, crow_end):
        value = tl.load(values_ptr + nnz * values_nnz_col_block_stride + block_offset * values_row_block_stride)

        # Compute the softmax value
        softmax_val = exp_val / sum_exp

        # Store the result
        tl.store(values_ptr + nnz * values_nnz_col_block_stride + block_offset * values_row_block_stride, softmax_val)

def _run_dense_rowspace_kernel(
    blocksize, values, crow_indices, col_indices, dense, output, max_grid
):
    grid = (max_grid[0], max_grid[1], max_grid[2])
    _bsr_strided_dense_rowspace_kernel[grid](
        blocksize[0], blocksize[1],
        values, values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        crow_indices, crow_indices.stride(0), crow_indices.stride(1),
        col_indices, col_indices.stride(0), col_indices.stride(1),
        dense, dense.stride(0), dense.stride(1), dense.stride(2), dense.stride(3),
        output, output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        tl.float32, True, 32
    )
