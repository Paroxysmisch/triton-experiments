import torch
import triton
import triton.language as tl

# Triton kernel for index_select_cat_fwd
@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,  # Pointer to the source tensor
    index_ptr,   # Pointer to the index tensor
    output_ptr,  # Pointer to the output tensor
    source_stride0,  # Stride for the first dimension of the source tensor
    source_stride1,  # Stride for the second dimension of the source tensor
    output_stride0,  # Stride for the first dimension of the output tensor
    output_stride1,  # Stride for the second dimension of the output tensor
    num_indices,     # Number of indices
    num_cols,        # Number of columns in the source tensor
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for indices
    BLOCK_SIZE_COL: tl.constexpr     # Block size for columns
):
    pid0 = tl.program_id(axis=0)  # Block ID for indices
    pid1 = tl.program_id(axis=1)  # Block ID for columns

    # Compute the range of indices and columns for this block
    index_start = pid0 * BLOCK_SIZE_INDEX
    index_end = min(index_start + BLOCK_SIZE_INDEX, num_indices)
    col_start = pid1 * BLOCK_SIZE_COL
    col_end = min(col_start + BLOCK_SIZE_COL, num_cols)

    # Iterate over the indices and columns
    for idx in range(index_start, index_end):
        source_row_idx = tl.load(index_ptr + idx)  # Load the index
        for col in range(col_start, col_end):
            # Compute the offsets for the source and output tensors
            source_offset = source_row_idx * source_stride0 + col * source_stride1
            output_offset = idx * output_stride0 + col * output_stride1

            # Load the data from the source tensor
            source_value = tl.load(source_ptr + source_offset)

            # Store the data into the output tensor
            tl.store(output_ptr + output_offset, source_value)

# Function to compute the grid dimensions
def grid(num_indices, num_cols, BLOCK_SIZE_INDEX, BLOCK_SIZE_COL):
    return (triton.cdiv(num_indices, BLOCK_SIZE_INDEX), triton.cdiv(num_cols, BLOCK_SIZE_COL))

# Wrapper function for the index_select_cat_fwd
def index_select_cat_fwd(source, index, output):
    assert source.dim() == 2, "Source tensor must be 2-dimensional"
    assert index.dim() == 1, "Index tensor must be 1-dimensional"
    assert source.is_cuda, "Source tensor must be on CUDA device"
    assert index.is_cuda, "Index tensor must be on CUDA device"

    num_rows, num_cols = source.shape
    num_indices = index.shape[0]

    if num_indices > num_rows:
        print("Warning: Number of indices exceeds the number of rows in the source. Indices will be truncated.")
        num_indices = num_rows

    # Allocate the output tensor
    output = torch.empty((num_indices, num_cols), dtype=source.dtype, device=source.device)

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 128

    # Compute the grid dimensions
    grid_dim = grid(num_indices, num_cols, BLOCK_SIZE_INDEX, BLOCK_SIZE_COL)

    # Launch the kernel
    index_select_cat_fwd_kernel[grid_dim](
        source_ptr=source,
        index_ptr=index,
        output_ptr=output,
        source_stride0=source.stride(0),
        source_stride1=source.stride(1),
        output_stride0=output.stride(0),
        output_stride1=output.stride(1),
        num_indices=num_indices,
        num_cols=num_cols,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return output

# Example usage
if __name__ == "__main__":
    source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device="cuda")
    index = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    output = index_select_cat_fwd(source, index, None)
    print(output)
