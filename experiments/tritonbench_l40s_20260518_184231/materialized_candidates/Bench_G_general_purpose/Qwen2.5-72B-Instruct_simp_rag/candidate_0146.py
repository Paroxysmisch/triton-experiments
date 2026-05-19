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

    # Compute the row and column block indices
    row_block_id = block_id // (k // TILE_K)
    col_block_id = block_id % (k // TILE_K)

    # Load the values, crow_indices, and col_indices
    values = tl.load(values_ptr + row_block_id * values_nnz_stride + col_block_id * values_col_block_stride)
    crow_indices = tl.load(crow_indices_ptr + row_block_id * crow_indices_stride)
    col_indices = tl.load(col_indices_ptr + col_block_id * col_indices_stride)

    # Load the sub-matrices from mat1 and mat2
    mat1 = tl.load(mat1_ptr + row_block_id * mat1_tiled_row_stride + col_block_id * mat1_tiled_col_stride)
    mat2 = tl.load(mat2_ptr + row_block_id * mat2_tiled_row_stride + col_block_id * mat2_tiled_col_stride)

    # Compute the dot product
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
    for i in range(TILE_K):
        acc += tl.dot(mat1, mat2, allow_tf32=allow_tf32)

    # Apply alpha and beta
    if IS_BETA_ZERO:
        acc = alpha * acc
    else:
        acc = beta * values + alpha * acc

    # Store the result
    tl.store(values_ptr + row_block_id * values_nnz_stride + col_block_id * values_col_block_stride, acc)

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

    # Compute the row and column block indices
    row_block_id = block_id // (BLOCKSIZE_COL // GROUP_SIZE_ROW)
    col_block_id = block_id % (BLOCKSIZE_COL // GROUP_SIZE_ROW)

    # Load the values, crow_indices, and col_indices
    values = tl.load(values_ptr + row_block_id * values_nnz_stride + col_block_id * values_col_block_stride)
    crow_indices = tl.load(crow_indices_ptr + row_block_id * crow_indices_stride)
    col_indices = tl.load(col_indices_ptr + col_block_id * col_indices_stride)

    # Load the sub-matrices from dense
    dense = tl.load(dense_ptr + row_block_id * dense_tiled_row_stride + col_block_id * dense_tiled_col_stride)

    # Compute the dot product
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)
    for i in range(GROUP_SIZE_ROW):
        acc += tl.dot(values, dense, allow_tf32=allow_tf32)

    # Store the result
    tl.store(output_ptr + row_block_id * output_tiled_row_stride + col_block_id * output_tiled_col_stride, acc)

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

    # Compute the row and column block indices
    row_block_id = block_id // (col_block // TILE)
    col_block_id = block_id % (col_block // TILE)

    # Load the values and crow_indices
    values = tl.load(values_ptr + row_block_id * values_row_block_stride + col_block_id * values_nnz_col_block_stride)
    crow_indices = tl.load(crow_indices_ptr + row_block_id * crow_indices_stride)

    # Compute the maximum value in the row
    max_val = tl.max(values, axis=0)

    # Compute the exponentials
    exp_values = tl.exp(values - max_val)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_values, axis=0)

    # Compute the softmax values
    softmax_values = exp_values / sum_exp

    # Store the result
    tl.store(values_ptr + row_block_id * values_row_block_stride + col_block_id * values_nnz_col_block_stride, softmax_values)

def _run_dense_rowspace_kernel(
    blocksize, values, crow_indices, col_indices, dense, output, max_grid
):
    grid = max_grid or (values.size(0),)
    _bsr_strided_dense_rowspace_kernel[grid](
        blocksize[0], blocksize[1],
        values.data_ptr(), values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        crow_indices.data_ptr(), crow_indices.stride(0), crow_indices.stride(1),
        col_indices.data_ptr(), col_indices.stride(0), col_indices.stride(1),
        dense.data_ptr(), dense.stride(0), dense.stride(1), dense.stride(2), dense.stride(3),
        output.data_ptr(), output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        tl.float32, True, 16
    )

def _run_sampled_addmm_kernel(
    alpha, beta, is_beta_zero,
    blocksize, k, tile_k,
    values, crow_indices, col_indices,
    mat1, mat2,
    max_grid
):
    grid = max_grid or (values.size(0),)
    _sampled_addmm_kernel[grid](
        alpha, beta, is_beta_zero,
        blocksize[0], blocksize[1],
        k, tile_k,
        values.data_ptr(), values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        crow_indices.data_ptr(), crow_indices.stride(0), crow_indices.stride(1),
        col_indices.data_ptr(), col_indices.stride(0), col_indices.stride(1),
        mat1.data_ptr(), mat1.stride(0), mat1.stride(1), mat1.stride(2), mat1.stride(3),
        mat2.data_ptr(), mat2.stride(0), mat2.stride(1), mat2.stride(2), mat2.stride(3),
        tl.float32, True
    )
