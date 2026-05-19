import triton
import triton.language as tl
import torch
import warnings

@triton.jit
def index_select_cat_fwd_kernel(
    source_ptr,    # Pointer to source tensor
    index_ptr,     # Pointer to index tensor
    output_ptr,    # Pointer to output tensor
    n_indices,     # Number of indices
    n_cols,        # Number of columns
    source_stride_0, # Stride for source tensor rows
    source_stride_1, # Stride for source tensor columns
    output_stride_0, # Stride for output tensor rows
    output_stride_1, # Stride for output tensor columns
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for indices
    BLOCK_SIZE_COL: tl.constexpr,    # Block size for columns
):
    # Program ID for index and column blocks
    pid0 = tl.program_id(0)  # Index block
    pid1 = tl.program_id(1)  # Column block
    
    # Starting positions
    index_start = pid0 * BLOCK_SIZE_INDEX
    col_start = pid1 * BLOCK_SIZE_COL
    
    # Generate offsets for indices and columns
    index_offsets = index_start + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_COL)
    
    # Create masks for bounds checking
    index_mask = index_offsets < n_indices
    col_mask = col_offsets < n_cols
    
    # Load indices for this block
    indices = tl.load(index_ptr + index_offsets, mask=index_mask)
    
    # For each valid index and column combination
    for idx in range(BLOCK_SIZE_INDEX):
        if idx < n_indices:
            # Get source row offset
            source_row = indices[idx] * source_stride_0
            # Get output row offset
            output_row = (index_start + idx) * output_stride_0
            
            # Load and store data
            x = tl.load(
                source_ptr + source_row + col_offsets * source_stride_1,
                mask=col_mask
            )
            tl.store(
                output_ptr + output_row + col_offsets * output_stride_1,
                x,
                mask=col_mask
            )

def index_select_cat_fwd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Select and concatenate rows from source tensor based on indices.
    
    Args:
        source (torch.Tensor): 2D input tensor
        index (torch.Tensor): 1D tensor containing indices to select
        
    Returns:
        torch.Tensor: Selected and concatenated tensor
    """
    assert source.dim() == 2, "Source tensor must be 2-dimensional"
    assert index.dim() == 1, "Index tensor must be 1-dimensional"
    assert source.is_cuda and index.is_cuda, "Both tensors must be on CUDA"
    
    if index.numel() > source.size(0):
        warnings.warn(
            f"Number of indices ({index.numel()}) exceeds number of rows in source ({source.size(0)}). "
            "Indices will be truncated."
        )
        index = index[:source.size(0)]
    
    # Get dimensions
    n_indices = index.numel()
    n_cols = source.size(1)
    
    # Create output tensor
    output = torch.empty(
        (n_indices, n_cols),
        device=source.device,
        dtype=source.dtype
    )
    
    # Get strides
    source_stride_0 = source.stride(0)
    source_stride_1 = source.stride(1)
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    
    # Define block sizes
    BLOCK_SIZE_INDEX = 16
    BLOCK_SIZE_COL = 32
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(n_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(n_cols, BLOCK_SIZE_COL)
    )
    
    # Launch kernel
    index_select_cat_fwd_kernel[grid](
        source.data_ptr(),
        index.data_ptr(),
        output.data_ptr(),
        n_indices,
        n_cols,
        source_stride_0,
        source_stride_1,
        output_stride_0,
        output_stride_1,
        BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL,
    )
    
    return output
