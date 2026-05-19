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
    num_pid_m = tl.cdiv(BLOCKSIZE_ROW, BLOCKSIZE_ROW)
    num_pid_n = tl.cdiv(BLOCKSIZE_COL, BLOCKSIZE_COL)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_m = (pid % num_pid_in_batch) // num_pid_n
    pid_n = (pid % num_pid_in_batch) % num_pid_n

    offs_m = pid_m * BLOCKSIZE_ROW + tl.arange(0, BLOCKSIZE_ROW)
    offs_n = pid_n * BLOCKSIZE_COL + tl.arange(0, BLOCKSIZE_COL)
    offs_k = tl.arange(0, TILE_K)

    values_block_ptr = values_ptr + batch_id * values_batch_stride
    crow_indices_block_ptr = crow_indices_ptr + batch_id * crow_indices_batch_stride
    col_indices_block_ptr = col_indices_ptr + batch_id * col_indices_batch_stride
    mat1_block_ptr = mat1_ptr + batch_id * mat1_batch_stride
    mat2_block_ptr = mat2_ptr + batch_id * mat2_batch_stride

    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    for k_start in range(0, k, TILE_K):
        k_end = min(k, k_start + TILE_K)
        for i in range(k_start, k_end):
            row_start = tl.load(crow_indices_block_ptr + i * crow_indices_stride)
            row_end = tl.load(crow_indices_block_ptr + (i + 1) * crow_indices_stride)
            for j in range(row_start, row_end):
                col = tl.load(col_indices_block_ptr + j * col_indices_stride)
                value = tl.load(values_block_ptr + j * values_nnz_stride)
                mat1_val = tl.load(mat1_block_ptr + (offs_m * mat1_row_block_stride + i * mat1_tiled_row_stride))
                mat2_val = tl.load(mat2_block_ptr + (i * mat2_tiled_row_stride + offs_n * mat2_col_block_stride))
                acc += value * mat1_val * mat2_val

    if not IS_BETA_ZERO:
        output_ptr = out_ptr + batch_id * out_batch_stride + pid_m * out_row_block_stride + pid_n * out_col_block_stride
        out_val = tl.load(output_ptr)
        acc = acc * alpha + out_val * beta
    else:
        acc = acc * alpha

    output_ptr = out_ptr + batch_id * out_batch_stride + pid_m * out_row_block_stride + pid_n * out_col_block_stride
    tl.store(output_ptr, acc)

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
    num_pid_m = tl.cdiv(BLOCKSIZE_ROW, BLOCKSIZE_ROW)
    num_pid_n = tl.cdiv(BLOCKSIZE_COL, BLOCKSIZE_COL)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_m = (pid % num_pid_in_batch) // num_pid_n
    pid_n = (pid % num_pid_in_batch) % num_pid_n

    offs_m = pid_m * BLOCKSIZE_ROW + tl.arange(0, BLOCKSIZE_ROW)
    offs_n = pid_n * BLOCKSIZE_COL + tl.arange(0, BLOCKSIZE_COL)

    values_block_ptr = values_ptr + batch_id * values_batch_stride
    crow_indices_block_ptr = crow_indices_ptr + batch_id * crow_indices_batch_stride
    col_indices_block_ptr = col_indices_ptr + batch_id * col_indices_batch_stride
    dense_block_ptr = dense_ptr + batch_id * dense_batch_stride
    output_block_ptr = output_ptr + batch_id * output_batch_stride

    acc = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    for i in range(crow_indices_block_ptr[0], crow_indices_block_ptr[1]):
        col = col_indices_block_ptr[i]
        value = values_block_ptr[i * values_nnz_stride]
        dense_val = tl.load(dense_block_ptr + (offs_m * dense_row_block_stride + col * dense_col_block_stride))
        acc += value * dense_val

    output_ptr = output_block_ptr + (pid_m * output_row_block_stride + pid_n * output_col_block_stride)
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(row_block, TILE)
    num_pid_n = tl.cdiv(col_block, TILE)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_m = (pid % num_pid_in_batch) // num_pid_n
    pid_n = (pid % num_pid_in_batch) % num_pid_n

    offs_m = pid_m * TILE + tl.arange(0, TILE)
    offs_n = pid_n * TILE + tl.arange(0, TILE)

    values_block_ptr = values_ptr + batch_id * values_batch_stride
    crow_indices_block_ptr = crow_indices_ptr + batch_id * crow_indices_batch_stride

    max_val = -float('inf')
    for i in range(crow_indices_block_ptr[0], crow_indices_block_ptr[1]):
        value = values_block_ptr[i * values_nnz_col_block_stride]
        max_val = tl.max(max_val, value)

    sum_exp = 0.0
    for i in range(crow_indices_block_ptr[0], crow_indices_block_ptr[1]):
        value = values_block_ptr[i * values_nnz_col_block_stride]
        exp_val = tl.exp(value - max_val)
        sum_exp += exp_val

    for i in range(crow_indices_block_ptr[0], crow_indices_block_ptr[1]):
        value = values_block_ptr[i * values_nnz_col_block_stride]
        exp_val = tl.exp(value - max_val)
        softmax_val = exp_val / sum_exp
        values_block_ptr[i * values_nnz_col_block_stride] = softmax_val
