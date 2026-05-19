import triton
import triton.language as tl
import torch

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr, grad_source_ptr, index_ptr,
    stride_gout_row, stride_gout_col,
    stride_gsrc_row, stride_gsrc_col,
    stride_idx,
    num_indices, num_cols,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    # Program ID
    pid_i = tl.program_id(0)  # index dimension
    pid_j = tl.program_id(1)  # column dimension
    
    # Compute offsets for this block
    idx_offset = pid_i * BLOCK_SIZE_INDEX
    col_offset = pid_j * BLOCK_SIZE_COL
    
    # Create ranges for indices and columns
    idx_range = tl.arange(0, BLOCK_SIZE_INDEX)
    col_range = tl.arange(0, BLOCK_SIZE_COL)
    
    # Compute masks for valid indices and columns
    idx_mask = idx_offset + idx_range < num_indices
    col_mask = col_offset + col_range < num_cols
    
    # Load indices for this block
    indices = tl.load(index_ptr + idx_offset + idx_range * stride_idx, mask=idx_mask, other=0)
    
    # For each valid index and column, accumulate gradients
    for i in range(BLOCK_SIZE_INDEX):
        if idx_offset + i >= num_indices:
            break
            
        src_row = indices[i]
        
        for j in range(BLOCK_SIZE_COL):
            if col_offset + j >= num_cols:
                break
                
            # Load gradient from grad_output
            grad = tl.load(
                grad_output_ptr + (idx_offset + i) * stride_gout_row + (col_offset + j) * stride_gout_col,
                mask=idx_mask[i] & col_mask[j],
                other=0.0
            )
            
            # Accumulate gradient to grad_source using atomic_add
            tl.atomic_add(
                grad_source_ptr + src_row * stride_gsrc_row + (col_offset + j) * stride_gsrc_col,
                grad,
                mask=idx_mask[i] & col_mask[j]
            )

def index_select_cat_bwd(grad_output, grad_source, index):
    """
    Backward pass for index_select_cat operation.
    
    Args:
        grad_output: Gradient tensor of shape (num_indices, num_cols)
        grad_source: Gradient tensor to be updated, shape (num_rows, num_cols)
        index: Index tensor of shape (num_indices,)
    """
    # Input validation
    assert grad_output.dim() == 2, "grad_output must be 2D"
    assert grad_source.dim() == 2, "grad_source must be 2D"
    assert index.dim() == 1, "index must be 1D"
    assert grad_output.is_cuda and grad_source.is_cuda and index.is_cuda, "All tensors must be CUDA tensors"
    
    num_indices, num_cols = grad_output.shape
    
    # Block sizes for parallel processing
    BLOCK_SIZE_INDEX = 32
    BLOCK_SIZE_COL = 32
    
    # Grid configuration
    grid = (
        triton.cdiv(num_indices, BLOCK_SIZE_INDEX),
        triton.cdiv(num_cols, BLOCK_SIZE_COL)
    )
    
    # Launch kernel
    index_select_cat_bwd_kernel[grid](
        grad_output_ptr=grad_output.data_ptr(),
        grad_source_ptr=grad_source.data_ptr(),
        index_ptr=index.data_ptr(),
        stride_gout_row=grad_output.stride(0),
        stride_gout_col=grad_output.stride(1),
        stride_gsrc_row=grad_source.stride(0),
        stride_gsrc_col=grad_source.stride(1),
        stride_idx=index.stride(0),
        num_indices=num_indices,
        num_cols=num_cols,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return grad_source
