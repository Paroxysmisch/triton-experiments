import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,  # pointer to the source tensor
    index_ptr,   # pointer to the index tensor
    output_ptr,  # pointer to the output tensor
    num_indices, # number of indices
    num_cols,    # number of columns in the source tensor
    source_stride0, # stride for source tensor's rows
    BLOCK_SIZE_INDEX: tl.constexpr, # block size for indices
    BLOCK_SIZE_COL: tl.constexpr    # block size for columns
):
    # Program ids for indices and columns
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Compute the starting index for this block
    row_start = pid0 * BLOCK_SIZE_INDEX
    col_start = pid1 * BLOCK_SIZE_COL

    # Create ranges for indices and columns
    rows = row_start + tl.arange(0, BLOCK_SIZE_INDEX)
    cols = col_start + tl.arange(0, BLOCK_SIZE_COL)

    # Load the indices for this block
    indices = tl.load(index_ptr + rows, mask=rows < num_indices, other=0)

    # Compute offsets for source and output tensors
    source_offsets = indices[:, None] * source_stride0 + cols[None, :]
    output_offsets = rows[:, None] * num_cols + cols[None, :]

    # Load data from the source tensor
    data = tl.load(source_ptr + source_offsets, mask=(rows[:, None] < num_indices) & (cols[None, :] < num_cols), other=0)

    # Store the selected data into the output tensor
    tl.store(output_ptr + output_offsets, data, mask=(rows[:, None] < num_indices) & (cols[None, :] < num_cols))

def index_select_cat_fwd(source, index):
    assert source.is_cuda and index.is_cuda, "source and index must be CUDA tensors"
    assert source.dim() == 2, "source must be a 2D tensor"
    assert index.dim() == 1, "index must be a 1D tensor"

    num_indices = index.size(0)
    num_rows, num_cols = source.size()

    if num_indices > num_rows:
        print("Warning: Number of indices exceeds number of rows in the source. Truncating indices.")
        num_indices = num_rows

    # Allocate output tensor
    output = torch.empty((num_indices, num_cols), device=source.device, dtype=source.dtype)

    # Extract strides for source tensor dimension traversal
    source_stride0 = source.stride(0)

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 128

    # Compute grid dimensions
    grid = (
        triton.cdiv(num_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(num_cols, BLOCK_SIZE_COL)
    )

    # Launch the Triton kernel
    index_select_cat_fwd_kernel[grid](
        source_ptr=source,
        index_ptr=index,
        output_ptr=output,
        num_indices=num_indices,
        num_cols=num_cols,
        source_stride0=source_stride0,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return output
