import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, index_ptr, output_ptr,
    source_row_stride, output_row_stride,
    n_indices, n_cols,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Compute the offsets for this thread block
    pid_index = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Compute the starting index and column for this thread block
    index_start = pid_index * BLOCK_SIZE_INDEX
    col_start = pid_col * BLOCK_SIZE_COL

    # Create offsets for this thread within the block
    index_offsets = tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = tl.arange(0, BLOCK_SIZE_COL)

    # Compute global index and column positions
    indices = index_start + index_offsets
    cols = col_start + col_offsets

    # Mask for bounds checking
    index_mask = indices < n_indices
    col_mask = cols < n_cols

    # Load indices
    indices = tl.load(index_ptr + indices, mask=index_mask, other=0)

    # Compute source and output offsets
    source_offsets = indices[:, None] * source_row_stride + cols[None, :]
    output_offsets = indices[:, None] * output_row_stride + cols[None, :]

    # Load from source and store to output
    for i in range(BLOCK_SIZE_INDEX):
        for j in range(BLOCK_SIZE_COL):
            if index_mask[i] and col_mask[j]:
                value = tl.load(source_ptr + source_offsets[i, j])
                tl.store(output_ptr + output_offsets[i, j], value)

# Wrapper function
def index_select_cat_fwd(source, index):
    n_indices = index.shape[0]
    n_rows, n_cols = source.shape

    # Allocate output tensor
    output = torch.empty((n_indices, n_cols), dtype=source.dtype, device=source.device)

    # Define block sizes
    BLOCK_SIZE_INDEX = 16
    BLOCK_SIZE_COL = 32

    # Calculate grid dimensions
    grid = (
        triton.cdiv(n_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(n_cols, BLOCK_SIZE_COL)
    )

    # Launch kernel
    index_select_cat_fwd_kernel[grid](
        source, index, output,
        source.stride(0), output.stride(0),
        n_indices, n_cols,
        BLOCK_SIZE_INDEX, BLOCK_SIZE_COL
    )

    return output
