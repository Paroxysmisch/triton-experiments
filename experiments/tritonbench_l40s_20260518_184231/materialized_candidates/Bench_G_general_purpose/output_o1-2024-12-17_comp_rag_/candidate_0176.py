import warnings
import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, 
    index_ptr,
    out_ptr,
    M, 
    N, 
    idx_len,
    stride_s0, 
    stride_s1,
    stride_out0, 
    stride_out1,
    BLOCK_SIZE_INDEX: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr
):
    pid0 = tl.program_id(axis=0)  # Block id for rows (index dimension)
    pid1 = tl.program_id(axis=1)  # Block id for columns

    # Compute offsets for the index and columns
    i_offsets = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    c_offsets = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # Masks to guard memory operations
    i_mask = i_offsets < idx_len
    c_mask = c_offsets < N

    # Load row indices from index tensor
    rows = tl.load(index_ptr + i_offsets, mask=i_mask, other=0)

    # Compute source offsets
    row_offsets = rows * stride_s0
    col_offsets = c_offsets * stride_s1
    row_offsets = row_offsets[:, None]
    col_offsets = col_offsets[None, :]
    source_offsets = row_offsets + col_offsets

    # Compute output offsets
    out_i_offset = i_offsets * stride_out0
    out_c_offset = c_offsets * stride_out1
    out_i_offset = out_i_offset[:, None]
    out_c_offset = out_c_offset[None, :]
    out_offsets = out_i_offset + out_c_offset

    # Load from source and store to out
    val = tl.load(source_ptr + source_offsets, mask=(i_mask[:, None] & c_mask[None, :]), other=0.0)
    tl.store(out_ptr + out_offsets, val, mask=(i_mask[:, None] & c_mask[None, :]))


def index_select_cat_fwd(source, index, block_size_index=128, block_size_col=128):
    if source.device.type != 'cuda' or index.device.type != 'cuda':
        raise ValueError("source and index must be CUDA tensors.")
    if source.ndim != 2:
        raise ValueError("source must be a 2D tensor.")
    if index.ndim != 1:
        raise ValueError("index must be a 1D tensor.")

    M, N = source.shape
    idx_len = index.numel()

    if idx_len > M:
        warnings.warn("Number of indices exceeds the number of rows. Truncating.")
        idx_len = M

    # Prepare output: shape will be [idx_len, N]
    out = torch.empty((idx_len, N), dtype=source.dtype, device=source.device)

    # Strides for source
    stride_s0 = source.stride(0)
    stride_s1 = source.stride(1)

    # Strides for output
    stride_out0 = out.stride(0)
    stride_out1 = out.stride(1)

    grid = (
        triton.cdiv(idx_len, block_size_index),
        triton.cdiv(N, block_size_col)
    )

    index_select_cat_fwd_kernel[grid](
        source, 
        index,
        out,
        M, 
        N, 
        idx_len,
        stride_s0, 
        stride_s1,
        stride_out0, 
        stride_out1,
        BLOCK_SIZE_INDEX=block_size_index, 
        BLOCK_SIZE_COL=block_size_col
    )
    return out
