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
    acc_dtype: tl.constexpr, allow_tf32: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Load row indices
    row_start = tl.load(crow_indices_ptr + pid * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + (pid + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCKSIZE_ROW, BLOCKSIZE_COL], dtype=acc_dtype)
    
    # Main computation loop
    for i in range(row_start, row_end):
        # Load column index
        col = tl.load(col_indices_ptr + i * col_indices_stride)
        
        # Load block from values
        block = tl.load(values_ptr + i * values_nnz_stride + 
                       tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
                       tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        
        # Load blocks from mat1 and mat2
        a = tl.load(mat1_ptr + pid * mat1_tiled_row_stride + 
                   col * mat1_tiled_col_stride +
                   tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat1_row_block_stride +
                   tl.arange(0, BLOCKSIZE_COL)[None, :] * mat1_col_block_stride)
        
        b = tl.load(mat2_ptr + col * mat2_tiled_row_stride +
                   tl.arange(0, BLOCKSIZE_ROW)[:, None] * mat2_row_block_stride +
                   tl.arange(0, BLOCKSIZE_COL)[None, :] * mat2_col_block_stride)
        
        # Compute matrix multiplication
        acc += alpha * (a * b) + (0 if IS_BETA_ZERO else beta * block)
    
    # Store result
    tl.store(values_ptr + pid * values_nnz_stride +
             tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride,
             acc)

@triton.jit
def _bsr_strided_dense_rowspace_kernel(
    BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr,
    values_ptr, values_batch_stride, values_nnz_stride,
    values_row_block_stride, values_col_block_stride,
    crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride,
    col_indices_ptr, col_indices_batch_stride, col_indices_stride,
    dense_ptr, dense_batch_stride, dense_tiled_row_stride,
    dense_tiled_col_stride, dense_row_block_stride, dense_col_block_stride,
    output_ptr, output_batch_stride, output_tiled_row_stride,
    output_tiled_col_stride, output_row_block_stride, output_col_block_stride,
    acc_dtype: tl.constexpr, allow_tf32: tl.constexpr,
    GROUP_SIZE_ROW: tl.constexpr):
    
    # Get program ID and initialize accumulator
    pid = tl.program_id(0)
    row = pid // GROUP_SIZE_ROW
    
    # Load row boundaries
    row_start = tl.load(crow_indices_ptr + row * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + (row + 1) * crow_indices_stride)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCKSIZE_ROW, BLOCKSIZE_COL], dtype=acc_dtype)
    
    # Main computation loop
    for i in range(row_start, row_end):
        # Load column index and values
        col = tl.load(col_indices_ptr + i * col_indices_stride)
        
        sparse_block = tl.load(values_ptr + i * values_nnz_stride +
                             tl.arange(0, BLOCKSIZE_ROW)[:, None] * values_row_block_stride +
                             tl.arange(0, BLOCKSIZE_COL)[None, :] * values_col_block_stride)
        
        dense_block = tl.load(dense_ptr + col * dense_tiled_row_stride +
                            tl.arange(0, BLOCKSIZE_ROW)[:, None] * dense_row_block_stride +
                            tl.arange(0, BLOCKSIZE_COL)[None, :] * dense_col_block_stride)
        
        # Multiply and accumulate
        acc += sparse_block * dense_block
    
    # Store result
    tl.store(output_ptr + row * output_tiled_row_stride +
             tl.arange(0, BLOCKSIZE_ROW)[:, None] * output_row_block_stride +
             tl.arange(0, BLOCKSIZE_COL)[None, :] * output_col_block_stride,
             acc)

@triton.jit
def _bsr_softmax_kernel(
    crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride,
    values_ptr, values_batch_stride, values_row_block_stride,
    values_nnz_col_block_stride, row_block, col_block,
    MAX_ROW_NNZ: tl.constexpr, TILE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Load row indices
    row_start = tl.load(crow_indices_ptr + pid * crow_indices_stride)
    row_end = tl.load(crow_indices_ptr + (pid + 1) * crow_indices_stride)
    nnz = row_end - row_start
    
    # Initialize max value
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    
    # Find max value
    for i in range(row_start, row_end):
        val = tl.load(values_ptr + i * values_nnz_col_block_stride)
        max_val = tl.maximum(max_val, val)
    
    # Compute exponentials and sum
    exp_sum = tl.zeros([1], dtype=tl.float32)
    for i in range(row_start, row_end):
        val = tl.load(values_ptr + i * values_nnz_col_block_stride)
        exp_val = tl.exp(val - max_val)
        tl.store(values_ptr + i * values_nnz_col_block_stride, exp_val)
        exp_sum += exp_val
    
    # Normalize
    for i in range(row_start, row_end):
        val = tl.load(values_ptr + i * values_nnz_col_block_stride)
        normalized = val / exp_sum
        tl.store(values_ptr + i * values_nnz_col_block_stride, normalized)

def sampled_addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor,
                 *, beta=1.0, alpha=1.0, out: Optional[torch.Tensor] = None,
                 skip_checks: bool = False,
                 max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None):
    # Get shapes and validate inputs
    batch_size = input.size(0)
    blocksize = input.size(-1)
    
    # Launch kernel
    grid = (batch_size,)
    _sampled_addmm_kernel[grid](
        alpha=alpha, beta=beta,
        IS_BETA_ZERO=(beta == 0),
        BLOCKSIZE_ROW=blocksize,
        BLOCKSIZE_COL=blocksize,
        k=mat1.size(1),
        TILE_K=32,
        values_ptr=input.data_ptr(),
        values_batch_stride=input.stride(0),
        values_nnz_stride=input.stride(1),
        values_row_block_stride=input.stride(-2),
        values_col_block_stride=input.stride(-1),
        crow_indices_ptr=input.crow_indices().data_ptr(),
        crow_indices_batch_stride=input.crow_indices().stride(0),
        crow_indices_stride=input.crow_indices().stride(1),
        col_indices_ptr=input.col_indices().data_ptr(),
        col_indices_batch_stride=input.col_indices().stride(0),
        col_indices_stride=input.col_indices().stride(1),
        mat1_ptr=mat1.data_ptr(),
        mat1_batch_stride=mat1.stride(0),
        mat1_tiled_row_stride=mat1.stride(1),
        mat1_tiled_col_stride=mat1.stride(2),
        mat1_row_block_stride=mat1.stride(-2),
        mat1_col_block_stride=mat1.stride(-1),
        mat2_ptr=mat2.data_ptr(),
        mat2_batch_stride=mat2.stride(0),
        mat2_tiled_row_stride=mat2.stride(1),
        mat2_tiled_col_stride=mat2.stride(2),
        mat2_row_block_stride=mat2.stride(-2),
        mat2_col_block_stride=mat2.stride(-1),
        acc_dtype=tl.float32,
        allow_tf32=True
    )
    return input
