import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source, output, M, N, index, index_len, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Get program ids for the 2D grid
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Calculate row and column offsets
    row_offsets = pid_x * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)[:, None]
    row_mask = row_offsets < index_len
    col_offsets = pid_y * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < N

    # Compute valid block mask
    block_mask = row_mask & col_mask

    # Load indices and compute offsets
    indices = tl.load(index + row_offsets, mask=row_mask, other=0)
    source_offsets = indices * N + col_offsets[None, :]
    output_offsets = row_offsets * N + col_offsets[None, :]

    # Load selected rows from source and store them in output
    selected_rows = tl.load(source + source_offsets, mask=block_mask, other=0.0)
    tl.store(output + output_offsets, selected_rows, mask=block_mask)

def index_select_cat_fwd(source, index):
    assert index.ndim == 1, "Index should be a 1D tensor"
    assert all((i >= 0 and i < source.size(0)) for i in index), "Index out of range"

    M, N = source.shape
    index_len = index.numel()

    # Prepare the output tensor
    output_shape = (index_len, N)
    output = torch.empty(output_shape, dtype=source.dtype, device=source.device)

    # Define grid size based on blocks
    BLOCK_SIZE_INDEX = 128  # Example block size for index dimension
    BLOCK_SIZE_COL = 128    # Example block size for column dimension

    grid = lambda meta: (
        triton.cdiv(index_len, meta["BLOCK_SIZE_INDEX"]),
        triton.cdiv(N, meta["BLOCK_SIZE_COL"]),
    )

    # Call the kernel with the calculated grid
    index_select_cat_fwd_kernel[grid](
        source, output, M, N, index, index_len,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return output
