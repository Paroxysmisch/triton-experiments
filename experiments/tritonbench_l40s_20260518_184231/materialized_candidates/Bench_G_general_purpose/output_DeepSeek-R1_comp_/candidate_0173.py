import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr, index_ptr, output_ptr,
    M,  # Number of rows in the source
    D,  # Number of columns in the source
    num_indices,  # Number of indices (truncated if necessary)
    stride_src_row, stride_src_col,  # Strides of the source tensor
    stride_idx,  # Stride of the index tensor
    stride_out_row, stride_out_col,  # Strides of the output tensor
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for indices dimension
    BLOCK_SIZE_COL: tl.constexpr,  # Block size for columns dimension
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Starting indices for the current block
    i_start = pid0 * BLOCK_SIZE_INDEX
    j_start = pid1 * BLOCK_SIZE_COL

    # Offsets for indices and columns in the current block
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_INDEX)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_COL)

    # Masks to avoid out-of-bounds access
    mask_i = i_offsets < num_indices
    mask_j = j_offsets < D
    mask = mask_i[:, None] & mask_j[None, :]

    # Load indices from the index tensor
    idx_offsets = i_offsets
    indices = tl.load(
        index_ptr + idx_offsets * stride_idx,
        mask=mask_i,
        other=0  # Default value if out of bounds (masked)
    )

    # Check if indices are within valid range [0, M-1]
    mask_idx_valid = (indices >= 0) & (indices < M)
    mask = mask & mask_idx_valid[:, None]

    # Calculate pointers for source rows
    src_row_ptrs = source_ptr + indices[:, None] * stride_src_row
    # Calculate pointers for source elements in the current block
    src_ptrs = src_row_ptrs + j_offsets[None, :] * stride_src_col

    # Load data from source
    data = tl.load(src_ptrs, mask=mask, other=0)

    # Calculate pointers for output
    out_row_ptrs = output_ptr + i_offsets[:, None] * stride_out_row
    out_ptrs = out_row_ptrs + j_offsets[None, :] * stride_out_col

    # Store the selected data into the output tensor
    tl.store(out_ptrs, data, mask=mask)

def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor):
    # Check input validity
    assert source.is_cuda and index.is_cuda, "Inputs must be CUDA tensors"
    assert source.dim() == 2, "Source must be a 2D tensor"
    assert index.dim() == 1, "Index must be a 1D tensor"

    M, D = source.shape
    num_indices = index.size(0)

    # Truncate indices if necessary
    if num_indices > M:
        print("Warning: Number of indices exceeds source rows; truncating to source row count.")
        num_indices = M
        index = index[:num_indices]

    # Initialize output tensor
    output = torch.empty((num_indices, D), device=source.device, dtype=source.dtype)

    # Strides for the source tensor
    stride_src_row, stride_src_col = source.stride()
    # Stride for the index tensor (1D)
    stride_idx = index.stride(0)
    # Strides for the output tensor
    stride_out_row, stride_out_col = output.stride()

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 32

    # Grid dimensions
    grid = (
        triton.cdiv(num_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(D, BLOCK_SIZE_COL)
    )

    # Launch kernel
    index_select_cat_fwd_kernel[grid](
        source.data_ptr(),
        index.data_ptr(),
        output.data_ptr(),
        M,
        D,
        num_indices,
        stride_src_row,
        stride_src_col,
        stride_idx,
        stride_out_row,
        stride_out_col,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return output
