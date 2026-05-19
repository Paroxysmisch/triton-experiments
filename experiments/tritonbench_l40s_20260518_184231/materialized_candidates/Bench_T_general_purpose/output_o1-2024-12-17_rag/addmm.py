import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _addmm_sparse_kernel(
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
    batch_pid = tl.program_id(axis=1)
    row_block_pid = tl.program_id(axis=0)

    crow_indices_offset_ptr = (
        crow_indices_ptr
        + crow_indices_batch_stride * batch_pid
        + crow_indices_stride * row_block_pid
    )
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

    row_nnz = nnz_offset_next - nnz_offset
    if row_nnz == 0:
        return

    row_block_arange = tl.arange(0, BLOCKSIZE_ROW)
    col_block_arange = tl.arange(0, BLOCKSIZE_COL)

    values_block_ptrs = (
        values_ptr
        + values_batch_stride * batch_pid
        + values_nnz_stride * nnz_offset
        + values_row_block_stride * row_block_arange[:, None]
        + values_col_block_stride * col_block_arange[None, :]
    )

    col_index_nnz_ptr = (
        col_indices_ptr
        + col_indices_batch_stride * batch_pid
        + col_indices_stride * nnz_offset
    )

    mat1_block_ptrs = (
        mat1_ptr
        + mat1_batch_stride * batch_pid
        + mat1_tiled_row_stride * row_block_pid
        + mat1_row_block_stride * row_block_arange[:, None]
    )

    mat2_block_ptrs = (
        mat2_ptr
        + mat2_batch_stride * batch_pid
        + mat2_col_block_stride * col_block_arange[None, :]
    )

    k_tile_arange = tl.arange(0, TILE_K)
    for _ in range(row_nnz):
        acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

        col_block = tl.load(col_index_nnz_ptr)
        for k_tile in range(0, k, TILE_K):
            k_offsets = k_tile + k_tile_arange
            mask_k = k_offsets < k

            mat1_block = tl.load(
                mat1_block_ptrs
                + mat1_col_block_stride * k_offsets[None, :],
                mask=mask_k[None, :],
                other=0.0
            )

            mat2_block = tl.load(
                mat2_block_ptrs
                + mat2_tiled_col_stride * col_block
                + mat2_row_block_stride * k_offsets[:, None],
                mask=mask_k[:, None],
                other=0.0
            )

            acc_block += tl.dot(
                mat1_block,
                mat2_block,
                allow_tf32=allow_tf32,
                out_dtype=acc_dtype
            )

        if IS_BETA_ZERO:
            acc_block *= alpha
        else:
            acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)

        tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))
        values_block_ptrs += values_nnz_stride
        col_index_nnz_ptr += col_indices_stride

def _run_addmm_sparse_kernel(
    alpha: float,
    beta: float,
    is_beta_zero: bool,
    blocksize: Tuple[int, int],
    k: int,
    tile_k: int,
    values: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    allow_tf32 = (mat1.dtype == torch.float32)
    grid = (crow_indices.shape[-1], values.shape[0])
    if max_grid is not None:
        grid = (
            min(grid[0], max_grid[0]) if max_grid[0] is not None else grid[0],
            min(grid[1], max_grid[1]) if max_grid[1] is not None else grid[1],
        )

    stride_values_batch = values.stride(0)
    stride_values_nnz = values.stride(-3)
    stride_values_row_block = values.stride(-2)
    stride_values_col_block = values.stride(-1)
    stride_crow_indices_batch = crow_indices.stride(0)
    stride_crow_indices = crow_indices.stride(-1)
    stride_col_indices_batch = col_indices.stride(0)
    stride_col_indices = col_indices.stride(-1)

    BLOCKSIZE_ROW, BLOCKSIZE_COL = blocksize
    mat1_block_shape = mat1.stride()[-4:]
    mat2_block_shape = mat2.stride()[-4:]
    mat1_batch_stride = mat1.stride(0)
    mat2_batch_stride = mat2.stride(0)

    mat1_tiled_row_stride = mat1_block_shape[-4]
    mat1_tiled_col_stride = mat1_block_shape[-3]
    mat1_row_block_stride = mat1_block_shape[-2]
    mat1_col_block_stride = mat1_block_shape[-1]

    mat2_tiled_row_stride = mat2_block_shape[-4]
    mat2_tiled_col_stride = mat2_block_shape[-3]
    mat2_row_block_stride = mat2_block_shape[-
