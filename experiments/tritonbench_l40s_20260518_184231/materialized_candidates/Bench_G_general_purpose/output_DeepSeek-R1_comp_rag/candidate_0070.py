import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr,  # Pointer to gradient output tensor
    indices_ptr,      # Pointer to indices tensor
    grad_source_ptr,  # Pointer to gradient source tensor (to be updated)
    output_rows,      # Number of rows in grad_output
    source_rows,      # Number of rows in grad_source
    cols,             # Number of columns in tensors
    grad_output_row_stride,  # Stride between rows in grad_output
    grad_output_col_stride,  # Stride between columns in grad_output
    indices_stride,   # Stride in indices tensor
    grad_source_row_stride,  # Stride between rows in grad_source
    grad_source_col_stride,  # Stride between columns in grad_source
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for index dimension
    BLOCK_SIZE_COL: tl.constexpr,     # Block size for column dimension
):
    # 2D grid layout: [row_blocks, col_blocks]
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    # Create ranges for rows and columns
    row_start = row_block * BLOCK_SIZE_INDEX
    col_start = col_block * BLOCK_SIZE_COL

    # Offsets for rows and columns within block
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)

    # Create masks for valid elements
    row_mask = row_offsets < output_rows
    col_mask = col_offsets < cols
    full_mask = row_mask[:, None] & col_mask[None, :]

    # Load indices for current rows
    indices = tl.load(
        indices_ptr + row_offsets * indices_stride,
        mask=row_mask,
        other=0
    )

    # Load gradient values from output
    grad_ptrs = (
        grad_output_ptr + 
        row_offsets[:, None] * grad_output_row_stride +
        col_offsets[None, :] * grad_output_col_stride
    )
    grad_vals = tl.load(grad_ptrs, mask=full_mask, other=0.0)

    # Calculate target positions in source gradient
    source_rows = tl.full(indices.shape, source_rows, dtype=tl.int32)
    safe_indices = tl.where(indices < source_rows, indices, 0)

    # Prepare pointers for atomic add
    source_ptrs = (
        grad_source_ptr +
        safe_indices[:, None] * grad_source_row_stride +
        col_offsets[None, :] * grad_source_col_stride
    )

    # Atomic add with mask
    tl.atomic_add(
        source_ptrs,
        grad_vals,
        mask=full_mask & (indices[:, None] < source_rows)
    )

def index_select_cat_bwd(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    source_shape: tuple[int, int]
) -> torch.Tensor:
    # Validate input dimensions
    assert grad_output.is_cuda and indices.is_cuda
    assert grad_output.ndim == 2 and indices.ndim == 2
    assert indices.shape[0] == grad_output.shape[0]
    assert indices.shape[1] == 1  # Single index per row

    # Initialize gradient source tensor
    grad_source = torch.zeros(
        source_shape,
        dtype=grad_output.dtype,
        device=grad_output.device
    )

    # Get tensor dimensions
    output_rows, cols = grad_output.shape
    source_rows = source_shape[0]

    # Configure kernel launch parameters
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 128

    grid = (
        triton.cdiv(output_rows, BLOCK_SIZE_INDEX),
        triton.cdiv(cols, BLOCK_SIZE_COL),
    )

    # Launch kernel
    index_select_cat_bwd_kernel[grid](
        grad_output,
        indices,
        grad_source,
        output_rows,
        source_rows,
        cols,
        grad_output.stride(0),
        grad_output.stride(1),
        indices.stride(0),
        grad_source.stride(0),
        grad_source.stride(1),
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    return grad_source
