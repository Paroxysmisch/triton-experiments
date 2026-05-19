import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,
    output_ptr,
    M,  # Number of rows in the source tensor
    N,  # Number of columns in the source tensor
    index_ptr,
    index_len,  # Length of the index tensor after truncation
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # Determine the block's index and column IDs
    pid0 = tl.program_id(0)  # Index block ID
    pid1 = tl.program_id(1)  # Column block ID

    # Calculate the range of indices this block is responsible for
    index_offset = pid0 * BLOCK_SIZE_INDEX
    index_offsets = index_offset + tl.arange(0, BLOCK_SIZE_INDEX)
    index_mask = index_offsets < index_len

    # Load the row indices from the index tensor
    row_indices = tl.load(index_ptr + index_offsets, mask=index_mask, other=0)

    # Check if the loaded row indices are within the valid range
    row_mask = (row_indices >= 0) & (row_indices < M)

    # Calculate the range of columns this block is responsible for
    col_offset = pid1 * BLOCK_SIZE_COL
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < N

    # Create 2D masks by broadcasting dimensions
    index_mask_2d = index_mask[:, None]
    row_mask_2d = row_mask[:, None]
    col_mask_2d = col_mask[None, :]

    # Combined mask for valid elements (index, row, and column)
    combined_mask = index_mask_2d & row_mask_2d & col_mask_2d

    # Compute source and output offsets using broadcasting
    source_offsets = row_indices[:, None] * N + col_offsets[None, :]
    output_offsets = index_offsets[:, None] * N + col_offsets[None, :]

    # Load data from source and store to output
    data = tl.load(source_ptr + source_offsets, mask=combined_mask, other=0.0)
    tl.store(output_ptr + output_offsets, data, mask=combined_mask)


def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on CUDA and have correct dimensions
    assert source.is_cuda and index.is_cuda, "Tensors must be CUDA tensors"
    assert source.dim() == 2, "Source must be a 2D tensor"
    assert index.dim() == 1, "Index must be a 1D tensor"

    M, N = source.shape
    original_index_len = index.numel()

    # Truncate index if its length exceeds the number of rows in the source
    if original_index_len > M:
        print("Warning: Number of indices exceeds source rows; truncating.")
        index = index[:M]
        index_len = M
    else:
        index_len = original_index_len

    # Initialize output tensor
    output = torch.empty((index_len, N), dtype=source.dtype, device=source.device)

    # Define grid configuration function
    def grid(meta):
        return (
            triton.cdiv(index_len, meta['BLOCK_SIZE_INDEX']),
            triton.cdiv(N, meta['BLOCK_SIZE_COL']),
        )

    # Launch the Triton kernel with specified block sizes
    index_select_cat_fwd_kernel[grid](
        source, output, M, N, index, index_len,
        BLOCK_SIZE_INDEX=128, BLOCK_SIZE_COL=32
    )

    return output
