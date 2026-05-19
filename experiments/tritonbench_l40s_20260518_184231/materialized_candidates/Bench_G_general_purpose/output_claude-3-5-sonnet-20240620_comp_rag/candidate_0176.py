import torch
import triton
import triton.language as tl
import warnings

@triton.jit
def index_select_cat_fwd_kernel(
    # Pointers to tensors
    source_ptr,
    index_ptr, 
    output_ptr,
    # Dimensions
    num_rows,
    num_cols,
    num_indices,
    # Source tensor strides
    row_stride,
    col_stride,
    # Block sizes as compile-time constants
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    # Get program ID for this block
    pid_index = tl.program_id(0)  # index dimension
    pid_col = tl.program_id(1)    # column dimension
    
    # Calculate starting positions
    index_offset = pid_index * BLOCK_SIZE_INDEX
    col_offset = pid_col * BLOCK_SIZE_COL
    
    # Create ranges for this block
    index_range = index_offset + tl.arange(0, BLOCK_SIZE_INDEX)
    col_range = col_offset + tl.arange(0, BLOCK_SIZE_COL)
    
    # Create masks for bounds checking
    index_mask = index_range < num_indices
    col_mask = col_range < num_cols
    
    # Load indices for this block
    indices = tl.load(index_ptr + index_range, mask=index_mask, other=0)
    
    # Bounds check for loaded indices
    valid_indices = indices < num_rows
    combined_mask = index_mask[:, None] & col_mask[None, :] & valid_indices[:, None]
    
    # Calculate source offsets
    source_offsets = indices[:, None] * row_stride + col_range[None, :] * col_stride
    
    # Load from source tensor
    source_data = tl.load(
        source_ptr + source_offsets,
        mask=combined_mask,
        other=0.0
    )
    
    # Calculate output offsets
    output_offsets = index_range[:, None] * num_cols + col_range[None, :]
    
    # Store to output tensor
    tl.store(
        output_ptr + output_offsets,
        source_data,
        mask=combined_mask
    )

def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Select and concatenate rows from source tensor based on indices.
    
    Args:
        source (torch.Tensor): 2D input tensor
        index (torch.Tensor): 1D tensor containing row indices to select
        
    Returns:
        torch.Tensor: Selected and concatenated tensor
    """
    assert source.dim() == 2, "Source tensor must be 2D"
    assert index.dim() == 1, "Index tensor must be 1D"
    assert source.is_cuda and index.is_cuda, "Both tensors must be on CUDA"
    
    num_rows, num_cols = source.shape
    num_indices = index.numel()
    
    # Warn and truncate if too many indices
    if num_indices > num_rows:
        warnings.warn(f"Number of indices ({num_indices}) exceeds source rows ({num_rows})")
        index = index[:num_rows]
        num_indices = num_rows
    
    # Create output tensor
    output = torch.empty(
        (num_indices, num_cols),
        dtype=source.dtype,
        device=source.device
    )
    
    # Get strides for efficient memory access
    row_stride = source.stride(0)
    col_stride = source.stride(1)
    
    # Define block sizes
    BLOCK_SIZE_INDEX = 32
    BLOCK_SIZE_COL = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(num_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(num_cols, BLOCK_SIZE_COL)
    )
    
    # Launch kernel
    index_select_cat_fwd_kernel[grid](
        source.data_ptr(),
        index.data_ptr(),
        output.data_ptr(),
        num_rows,
        num_cols,
        num_indices,
        row_stride,
        col_stride,
        BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL
    )
    
    return output
