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
                mat1_block_ptrs + mat1_col_block_stride * k_offsets[None, :],
                mask=mask_k[None, :], other=0.0
            )
            mat2_block = tl.load(
                mat2_block_ptrs + mat2_tiled_col_stride * col_block + mat2_row_block_stride * k_offsets[:, None],
                mask=mask_k[:, None], other=0.0
            )
            acc_block += tl.dot(mat1_block, mat2_block, allow_tf32=allow_tf32, out_dtype=acc_dtype)

        if IS_BETA_ZERO:
            acc_block *= alpha
        else:
            acc_block = alpha * acc_block + beta * tl.load(values_block_ptrs)

        tl.store(values_block_ptrs, acc_block.to(values_ptr.dtype.element_ty))
        values_block_ptrs += values_nnz_stride
        col_index_nnz_ptr += col_indices_stride

def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def check(cond, msg):
        if not cond:
            raise ValueError(msg)

    def check_bsr_layout(t):
        check(t.layout == torch.sparse_bsr, "addmm(): only BSR sparse format is supported for the sparse argument.")

    def check_device(t, device):
        check(t.device == device and t.device.type == "cuda", "addmm(): all inputs must be on the same GPU device.")

    def check_mm_compatible_shapes(lhs, rhs):
        m, kl = lhs.shape[-2:]
        kr, n = rhs.shape[-2:]
        check(kl == kr, f"addmm(): mat1 and mat2 shapes cannot be multiplied ({m}x{kl} and {kr}x{n})")

    def check_dtype(t, dtype):
        check(t.dtype == dtype and t.dtype in (torch.half, torch.bfloat16, torch.float),
              "addmm(): inputs must be half, bfloat16, or float32")

    check_bsr_layout(input)
    check_device(mat1, input.device)
    check_device(mat2, input.device)
    check_mm_compatible_shapes(mat1, mat2)
    if input.dtype != torch.bool:
        check_dtype(mat1, input.dtype)
        check_dtype(mat2, input.dtype)
    else:
        check_dtype(mat1, mat2.dtype)

    if out is not None:
        check_bsr_layout(out)
        check_device(out, input.device)
        check_dtype(out, input.dtype)
        check(out.shape == input.shape and out._nnz() == input._nnz(),
              "addmm(): out tensor must match input shape and nnz")

    if out is None:
        out = input.to(mat1.dtype, copy=True)
    else:
        out.copy_(input)

    if out.numel() == 0 or out._nnz() == 0:
        return out

    blocksize = out.values().shape[-2:]
    k = mat1.size(-1)

    if alpha == 0.0 or k == 0:
        out.values().mul_(beta)
        return out

    def prepare_inputs(bsr, *dense_tensors):
        crow_indices = bsr.crow_indices().unsqueeze(0)
        col_indices = bsr.col_indices().unsqueeze(0)
        values = bsr.values().unsqueeze(0).contiguous()
        tensors = [t.unsqueeze(0).contiguous() for t in dense_tensors]
        batch_shape = torch.broadcast_shapes(values.shape[:-3], *(t.shape[:-2] for t in tensors))

        def broadcast_and_flatten(t, invariant_dims):
            return t.broadcast_to(batch_shape + invariant_dims).flatten(0, len(batch_shape)-1)

        crow_indices = broadcast_and_flatten(crow_indices, (-1,))
        col_indices = broadcast_and_flatten(col_indices, (-1,))
        values = broadcast_and_flatten(values, values.shape[-3:])
        tensors = [broadcast_and_flatten(t, t.shape[-2:]) for t in tensors]
        return crow_indices, col_indices, values, *tensors

    crow_indices, col_indices, values, mat1, mat2 = prepare_inputs(out, mat1, mat2)

    def tile_to_blocksize(t, blocksize):
        *rest, m, n = t.shape
        return t.view(rest + [m // blocksize[0], blocksize[0], n // blocksize[1], blocksize[1]]).transpose(-3, -2)

    mat1 = tile_to_blocksize(mat1, (blocksize[0], k))
    mat2 = tile_to_blocksize(mat2, (k, blocksize[1]))
    TILE_K = max(blocksize[0], blocksize[1])

    grid = (crow_indices.size(0) - 1, values.size(0))
    num_warps = 4 if blocksize[0]*blocksize[1] <= 256 else 8

    _addmm_kernel[grid](
        alpha, beta, beta == 0.0,
        blocksize[0], blocksize[1],
        k, TILE_K,
        values, values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        crow_indices, crow_indices.stride(0), crow_indices.stride(1),
        col_indices, col_indices.stride(0), col_indices.stride(1),
        mat1, mat1.stride(0), mat1.stride(1), mat1.stride(2), mat1.stride(3), mat1.stride(4),
        mat2, mat2.stride(0), mat2.stride(1), mat2.stride(2), mat2.stride(3), mat2.stride(4),
        acc_dtype=tl.float32 if values.dtype == torch.float32 else tl.float16,
        allow_tf32=True,
        num_warps=num_warps,
    )

    return out
