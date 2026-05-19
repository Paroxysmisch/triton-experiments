import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,    # Pointer to gradient of source tensor [N, D]
    grad_output_ptr,    # Pointer to gradient of output tensor [M, D]
    index_ptr,         # Pointer to index tensor [M]
    n_rows,            # Number of rows in source tensor (N)
    n_cols,            # Number of columns (D)
    n_indices,         # Number of indices (M)
    grad_source_stride_row,  # Row stride of grad_source
    grad_source_stride_col,  # Col stride of grad_source
    grad_output_stride_row,  # Row stride of grad_output
    grad_output_stride_col,  # Col stride of grad_output
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices per block
    BLOCK_SIZE_COL: tl.constexpr,    # Number of columns per block
):
    # Program ID
    pid_index = tl.program_id(0)  # Index dimension
    pid_col = tl.program_id(1)    # Column dimension

    # Generate offsets for the block
    index_offsets = pid_index * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    col_offsets = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    # Bounds checking masks
    mask_index = index_offsets < n_indices
    mask_cols = col_offsets < n_cols
    
    # Load indices for this block
    indices = tl.load(index_ptr + index_offsets, mask=mask_index)
    
    # For each valid index and column, accumulate gradients
    for idx in range(tl.static_range(0, BLOCK_SIZE_INDEX)):
        if idx < n_indices:
            # Get source row index
            source_idx = indices[idx]
            
            # Load grad_output values
            grad_out_row_offset = index_offsets[idx] * grad_output_stride_row
            grad_out_ptrs = grad_output_ptr + grad_out_row_offset + col_offsets * grad_output_stride_col
            grad_values = tl.load(grad_out_ptrs, mask=mask_cols)
            
            # Compute grad_source pointers
            grad_source_row_offset = source_idx * grad_source_stride_row
            grad_source_ptrs = grad_source_ptr + grad_source_row_offset + col_offsets * grad_source_stride_col
            
            # Atomic add to handle potential index collisions
            tl.atomic_add(grad_source_ptrs, grad_values, mask=mask_cols)

def index_select_cat_bwd(grad_output: torch.Tensor, index: torch.Tensor, source_size: tuple):
    """
    Backward pass for index_select_cat operation.
    
    Args:
        grad_output: Gradient tensor of shape [M, D]
        index: Index tensor of shape [M]
        source_size: Size of the source tensor [N, D]
    
    Returns:
        grad_source: Gradient tensor for source of shape [N, D]
    """
    assert grad_output.dim() == 2, "grad_output must be 2D"
    assert index.dim() == 1, "index must be 1D"
    assert len(source_size) == 2, "source_size must be 2D"
    assert grad_output.is_cuda and index.is_cuda, "All tensors must be CUDA tensors"
    
    n_rows, n_cols = source_size
    n_indices = index.shape[0]
    
    # Create output gradient tensor
    grad_source = torch.zeros(source_size, device=grad_output.device, dtype=grad_output.dtype)
    
    # Configure block sizes
    BLOCK_SIZE_INDEX = 32
    BLOCK_SIZE_COL = 32
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(n_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(n_cols, BLOCK_SIZE_COL)
    )
    
    # Launch kernel
    index_select_cat_bwd_kernel[grid](
        grad_source_ptr=grad_source,
        grad_output_ptr=grad_output,
        index_ptr=index,
        n_rows=n_rows,
        n_cols=n_cols,
        n_indices=n_indices,
        grad_source_stride_row=grad_source.stride(0),
        grad_source_stride_col=grad_source.stride(1),
        grad_output_stride_row=grad_output.stride(0),
        grad_output_stride_col=grad_output.stride(1),
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return grad_source
