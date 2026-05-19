import triton
import triton.language as tl

# Define the kernel
@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,  # Pointer to the source tensor
    index_ptr,   # Pointer to the index tensor
    output_ptr,  # Pointer to the output tensor
    source_stride0,  # Stride of the source tensor in the row dimension
    source_stride1,  # Stride of the source tensor in the column dimension
    output_stride0,  # Stride of the output tensor in the row dimension
    output_stride1,  # Stride of the output tensor in the column dimension
    num_indices,     # Number of indices
    num_cols,        # Number of columns in the source and output tensors
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for indices
    BLOCK_SIZE_COL: tl.constexpr     # Block size for columns
):
    # Get the block indices
    pid_index = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Compute the range of indices and columns for this block
    index_start = pid_index * BLOCK_SIZE_INDEX
    index_end = min(index_start + BLOCK_SIZE_INDEX, num_indices)
    col_start = pid_col * BLOCK_SIZE_COL
    col_end = min(col_start + BLOCK_SIZE_COL, num_cols)

    # Load the indices for this block
    indices = tl.arange(0, BLOCK_SIZE_INDEX)
    indices = tl.where(indices < (index_end - index_start), indices + index_start, 0)

    # Iterate over the columns
    for col in range(col_start, col_end):
        # Load the source data for this column
        source_offsets = indices * source_stride0 + col * source_stride1
        source_data = tl.load(source_ptr + source_offsets, mask=indices < (index_end - index_start))

        # Compute the output offsets
        output_offsets = (indices - index_start + pid_index * BLOCK_SIZE_INDEX) * output_stride0 + col * output_stride1

        # Store the data into the output tensor
        tl.store(output_ptr + output_offsets, source_data, mask=indices < (index_end - index_start))

# Define the grid function
def grid(meta):
    return (
        (meta['num_indices'] + meta['BLOCK_SIZE_INDEX'] - 1) // meta['BLOCK_SIZE_INDEX'],
        (meta['num_cols'] + meta['BLOCK_SIZE_COL'] - 1) // meta['BLOCK_SIZE_COL']
    )

# Define the wrapper function
def index_select_cat_fwd(source, index, output):
    assert source.dim() == 2, "Source tensor must be 2D"
    assert index.dim() == 1, "Index tensor must be 1D"
    assert output.dim() == 2, "Output tensor must be 2D"
    assert source.shape[1] == output.shape[1], "Source and output tensors must have the same number of columns"

    num_indices = index.shape[0]
    num_cols = source.shape[1]

    # Define the block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 32

    # Launch the kernel
    index_select_cat_fwd_kernel[grid({'num_indices': num_indices, 'num_cols': num_cols, 'BLOCK_SIZE_INDEX': BLOCK_SIZE_INDEX, 'BLOCK_SIZE_COL': BLOCK_SIZE_COL})](
        source.data_ptr(), index.data_ptr(), output.data_ptr(),
        source.stride(0), source.stride(1),
        output.stride(0), output.stride(1),
        num_indices, num_cols,
        BLOCK_SIZE_INDEX, BLOCK_SIZE_COL
    )
