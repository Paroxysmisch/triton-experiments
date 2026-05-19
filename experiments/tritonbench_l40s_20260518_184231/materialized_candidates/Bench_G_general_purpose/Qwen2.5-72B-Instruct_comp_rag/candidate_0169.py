import torch
import triton
import triton.language as tl

# Triton kernel for index select and concatenate
@triton.jit
def index_select_cat_fwd_kernel(
    source, output, M, N, index, index_len, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Get program ids for x and y axes
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    row_offsets = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)[:, None]
    row_mask = row_offsets < index_len
    col_offsets = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < N

    # Compute masks for blocks and output
    block_mask = row_mask & col_mask
    out_mask = row_mask & col_mask

    # Load indices and compute offsets
    indices = tl.load(index + row_offsets, mask=row_mask, other=0)
    source_off = indices * N + col_offsets[None, :]
    output_off = row_offsets * N + col_offsets[None, :]

    # Load selected input and store in output
    selected = tl.load(source + source_off, mask=block_mask, other=0.0)
    tl.store(output + output_off, selected, mask=out_mask)

# Wrapper function for the kernel
def index_select_cat_fwd(source, index):
    assert source.ndim == 2, "Source tensor must be 2-dimensional"
    assert index.ndim == 1, "Index tensor must be 1-dimensional"
    assert all((i >= 0 and i < source.size(0)) for i in index), "Index out of range"

    # Check if the number of indices exceeds the number of rows in the source
    if index.numel() > source.size(0):
        print("Warning: Number of indices exceeds the number of rows in the source. Indices will be truncated.")
        index = index[:source.size(0)]

    M = source.size(0)
    N = source.size(1)
    index_len = index.numel()
    out_shape = (index_len, N)
    output = torch.empty(out_shape, dtype=source.dtype, device=source.device)

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(index_len, meta["BLOCK_SIZE_INDEX"]),
        triton.cdiv(N, meta["BLOCK_SIZE_COL"]),
    )
    
    # Call the kernel with calculated grid
    index_select_cat_fwd_kernel[grid](source, output, M, N, index, index_len)

    return output
