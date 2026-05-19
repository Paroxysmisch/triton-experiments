import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, index_ptr, output_ptr,
    num_indices, num_cols,
    BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Define the block indices
    block_idx = tl.program_id(0)
    block_col = tl.program_id(1)

    # Calculate start index for this block
    start_idx = block_idx * BLOCK_SIZE_INDEX
    start_col = block_col * BLOCK_SIZE_COL

    # Create a range for this block
    range_idx = tl.arange(0, BLOCK_SIZE_INDEX) + start_idx
    range_col = tl.arange(0, BLOCK_SIZE_COL) + start_col

    # Mask to ensure we don't go out of bounds
    mask_idx = range_idx < num_indices
    mask_col = range_col < num_cols

    # Load indices from the index tensor
    indices = tl.load(index_ptr + range_idx, mask=mask_idx)

    # Calculate offsets for the source and output
    source_offsets = indices[:, None] * num_cols + range_col[None, :]
    output_offsets = range_idx[:, None] * num_cols + range_col[None, :]

    # Load data from the source tensor
    source_data = tl.load(source_ptr + source_offsets, mask=mask_idx[:, None] & mask_col[None, :])

    # Store the selected data into the output tensor
    tl.store(output_ptr + output_offsets, source_data, mask=mask_idx[:, None] & mask_col[None, :])

# Wrapper function to launch the kernel
def index_select_cat_fwd(source, index, output, BLOCK_SIZE_INDEX=128, BLOCK_SIZE_COL=128):
    num_indices = index.shape[0]
    num_cols = source.shape[1]

    # Calculate grid size
    grid = (
        (num_indices + BLOCK_SIZE_INDEX - 1) // BLOCK_SIZE_INDEX,
        (num_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    )

    # Launch the kernel
    index_select_cat_fwd_kernel[grid](
        source, index, output,
        num_indices, num_cols,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

# Example usage:
# Assuming source is a 2D CUDA tensor, index is a 1D CUDA tensor, and output is a pre-allocated 2D CUDA tensor
# index_select_cat_fwd(source, index, output)
