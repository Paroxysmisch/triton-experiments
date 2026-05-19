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
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_rows = tl.num_programs(axis=0)
    row = pid * BLOCKSIZE_ROW

    # Initialize pointers
    values_ptr += row * values_row_block_stride
    crow_indices_ptr += row * crow_indices_stride
    col_indices_ptr += row * col_indices_stride
    mat1_ptr += row * mat1_tiled_row_stride
    mat2_ptr += row * mat2_tiled_row_stride

    # Initialize accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    # Iterate over non-zero blocks
    while True:
        # Load crow index
        crow_idx = tl.load(crow_indices_ptr + pid)
        if crow_idx == 0:
            break

        # Load column index
        col_idx = tl.load(col_indices_ptr + pid)

        # Load values
        values = tl.load(values_ptr + col_idx * values_col_block_stride)

        # Load mat1 and mat2
        mat1 = tl.load(mat1_ptr + col_idx * mat1_tiled_col_stride)
        mat2 = tl.load(mat2_ptr + col_idx * mat2_tiled_col_stride)

        # Compute dot product
        acc += tl.dot(mat1, mat2, allow_tf32=allow_tf32)

        # Move to next non-zero block
        pid += num_rows

    # Apply alpha and beta
    if IS_BETA_ZERO:
        acc *= alpha
    else:
        acc = alpha * acc + beta * acc

    # Store result
    tl.store(values_ptr, acc)

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
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_rows = tl.num_programs(axis=0)
    row = pid * BLOCKSIZE_ROW

    # Initialize pointers
    values_ptr += row * values_row_block_stride
    crow_indices_ptr += row * crow_indices_stride
    col_indices_ptr += row * col_indices_stride
    dense_ptr += row * dense_tiled_row_stride
    output_ptr += row * output_tiled_row_stride

    # Initialize accumulator
    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    # Iterate over non-zero blocks
    while True:
        # Load crow index
        crow_idx = tl.load(crow_indices_ptr + pid)
        if crow_idx == 0:
            break

        # Load column index
        col_idx = tl.load(col_indices_ptr + pid)

        # Load values and dense
        values = tl.load(values_ptr + col_idx * values_col_block_stride)
        dense = tl.load(dense_ptr + col_idx * dense_tiled_col_stride)

        # Compute dot product
        acc += tl.dot(values, dense, allow_tf32=allow_tf32)

        # Move to next non-zero block
        pid += num_rows

    # Store result
    tl.store(output_ptr, acc)

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
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_rows = tl.num_programs(axis=0)
    row = pid * row_block

    # Initialize pointers
    values_ptr += row * values_row_block_stride
    crow_indices_ptr += row * crow_indices_stride

    # Initialize max and sum
    max_val = -float('inf')
    sum_val = 0.0

    # Iterate over non-zero blocks
    for i in range(MAX_ROW_NNZ):
        # Load crow index
        crow_idx = tl.load(crow_indices_ptr + pid * crow_indices_batch_stride + i)
        if crow_idx == 0:
            break

        # Load values
        values = tl.load(values_ptr + crow_idx * values_nnz_col_block_stride)

        # Compute max
        max_val = tl.max(max_val, values)

    # Compute sum
    for i in range(MAX_ROW_NNZ):
        # Load crow index
        crow_idx = tl.load(crow_indices_ptr + pid * crow_indices_batch_stride + i)
        if crow_idx == 0:
            break

        # Load values
        values = tl.load(values_ptr + crow_idx * values_nnz_col_block_stride)

        # Compute sum
        sum_val += tl.exp(values - max_val)

    # Compute softmax
    for i in range(MAX_ROW_NNZ):
        # Load crow index
        crow_idx = tl.load(crow_indices_ptr + pid * crow_indices_batch_stride + i)
        if crow_idx == 0:
            break

        # Load values
        values = tl.load(values_ptr + crow_idx * values_nnz_col_block_stride)

        # Compute softmax
        softmax_val = tl.exp(values - max_val) / sum_val

        # Store result
        tl.store(values_ptr + crow_idx * values_nnz_col_block_stride, softmax_val)

def _run_dense_rowspace_kernel(
    blocksize, values, crow_indices, col_indices, dense, output, max_grid
):
    BLOCKSIZE_ROW, BLOCKSIZE_COL = blocksize
    grid = (output.shape[0] // BLOCKSIZE_ROW, 1, 1)
    _bsr_strided_dense_rowspace_kernel[grid](
        BLOCKSIZE_ROW, BLOCKSIZE_COL,
        values.data_ptr(), values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        crow_indices.data_ptr(), crow_indices.stride(0), crow_indices.stride(1),
        col_indices.data_ptr(), col_indices.stride(0), col_indices.stride(1),
        dense.data_ptr(), dense.stride(0), dense.stride(1), dense.stride(2), dense.stride(3),
        output.data_ptr(), output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        tl.float32, True, 128
    )
