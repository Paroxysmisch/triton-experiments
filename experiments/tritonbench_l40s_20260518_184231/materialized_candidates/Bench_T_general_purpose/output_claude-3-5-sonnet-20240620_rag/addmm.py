import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def _addmm_kernel(
    alpha,
    beta,
    IS_BETA_ZERO: tl.constexpr,
    BLOCKSIZE_ROW: tl.constexpr,
    BLOCKSIZE_COL: tl.constexpr,
    k,
    TILE_K: tl.constexpr,
    input_ptr,
    mat1_ptr,
    mat2_ptr,
    out_ptr,
    input_batch_stride,
    mat1_batch_stride,
    mat2_batch_stride,
    out_batch_stride,
    acc_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    batch_pid = tl.program_id(axis=0)
    row_block_pid = tl.program_id(axis=1)

    # Load input, mat1, and mat2
    input_block_ptr = input_ptr + input_batch_stride * batch_pid
    mat1_block_ptr = mat1_ptr + mat1_batch_stride * batch_pid
    mat2_block_ptr = mat2_ptr + mat2_batch_stride * batch_pid
    out_block_ptr = out_ptr + out_batch_stride * batch_pid

    acc_block = tl.zeros((BLOCKSIZE_ROW, BLOCKSIZE_COL), dtype=acc_dtype)

    for k_tile in range(0, k, TILE_K):
        # Load blocks of mat1 and mat2
        mat1_block = tl.load(mat1_block_ptr + k_tile)
        mat2_block = tl.load(mat2_block_ptr + k_tile)

        # Perform matrix multiplication
        acc_block += tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32)

    if IS_BETA_ZERO:
        acc_block *= alpha
    else:
        acc_block += beta * tl.load(input_block_ptr)

    # Store the result
    tl.store(out_block_ptr, acc_block)

def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    out: Optional[torch.Tensor] = None,
    skip_checks: bool = False,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    f_name = "addmm"

    # Check tensor layouts and compatibility
    if not skip_checks:
        assert input.is_sparse, f"{f_name}(): input must be a sparse tensor."
        assert mat1.is_dense and mat2.is_dense, f"{f_name}(): mat1 and mat2 must be dense tensors."
        assert input.shape[-2] == mat1.shape[-1], f"{f_name}(): mat1 and mat2 shapes are incompatible."

    # Prepare output tensor
    if out is None:
        out = input.new_zeros((input.shape[0], mat2.shape[1]), dtype=mat1.dtype)

    # Launch the kernel
    grid = (input.shape[0], (input.shape[0] + BLOCKSIZE_ROW - 1) // BLOCKSIZE_ROW)
    _addmm_kernel[grid](
        alpha,
        beta,
        beta == 0.0,
        BLOCKSIZE_ROW,
        BLOCKSIZE_COL,
        mat1.shape[1],
        TILE_K,
        input,
        mat1,
        mat2,
        out,
        input.stride(0),
        mat1.stride(0),
        mat2.stride(0),
        out.stride(0),
        acc_dtype=tl.float32,
        allow_tf32=True,
    )

    return out
