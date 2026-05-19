import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, 
    index_ptr, 
    out_ptr,
    n_indices, 
    n_cols,
    stride_row_source, 
    stride_col_source,
    stride_row_out, 
    stride_col_out,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    row_offsets = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    row_mask = row_offsets < n_indices
    idx = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)

    mask = (col_offsets < n_cols) & row_mask
    source_offsets = idx * stride_row_source + col_offsets * stride_col_source
    vals = tl.load(source_ptr + source_offsets, mask=mask, other=0.0)

    out_offsets = row_offsets * stride_row_out + col_offsets * stride_col_out
    tl.store(out_ptr + out_offsets, vals, mask=mask)


def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Ensure index is truncated if it exceeds the number of rows in source
    if index.shape[0] > source.shape[0]:
        print("Warning: Number of indices exceeds number of rows in source, truncating.")
        index = index[: source.shape[0]]

    n_indices = index.shape[0]
    n_cols = source.shape[1]

    out = torch.empty((n_indices, n_cols), device=source.device, dtype=source.dtype)

    # Extract strides
    stride_row_source = source.stride(0)
    stride_col_source = source.stride(1)
    stride_row_out = out.stride(0)
    stride_col_out = out.stride(1)

    # Grid function
    def grid(meta):
        return (
            triton.cdiv(n_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(n_cols, meta["BLOCK_SIZE_COL"])
        )

    # Launch Triton kernel
    index_select_cat_fwd_kernel[grid](
        source, 
        index, 
        out,
        n_indices, 
        n_cols,
        stride_row_source, 
        stride_col_source,
        stride_row_out, 
        stride_col_out,
        BLOCK_SIZE_INDEX=128,
        BLOCK_SIZE_COL=128
    )
    return out
