import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_output_ptr,   # Pointer to the gradient output tensor
    index_ptr,         # Pointer to the index tensor
    grad_source_ptr,   # Pointer to the gradient source tensor
    num_indices,       # Number of indices
    num_cols,          # Number of columns in the tensors
    BLOCK_SIZE_INDEX: tl.constexpr,  # Block size for index dimension
    BLOCK_SIZE_COL: tl.constexpr     # Block size for column dimension
):
    # Compute the grid indices
    row_idx = tl.program_id(0) * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    col_idx = tl.program_id(1) * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # Mask to ensure we do not access out of bounds memory
    row_mask = row_idx < num_indices
    col_mask = col_idx < num_cols

    # Load indices
    indices = tl.load(index_ptr + row_idx, mask=row_mask, other=0)

    # Compute the memory locations for the gradient source
    grad_source_offset = indices[:, None] * num_cols + col_idx[None, :]

    # Load the gradient output and accumulate it to the gradient source
    grad_output = tl.load(grad_output_ptr + row_idx[:, None] * num_cols + col_idx[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    tl.atomic_add(grad_source_ptr + grad_source_offset, grad_output, mask=row_mask[:, None] & col_mask[None, :])


import torch

def index_select_cat_bwd(grad_output, index, grad_source):
    # Ensure the inputs are CUDA tensors
    assert grad_output.is_cuda and index.is_cuda and grad_source.is_cuda, "All tensors must be CUDA tensors"
    assert grad_output.dim() == 2 and index.dim() == 1 and grad_source.dim() == 2, "Input dimensions are not compatible"
    
    # Get dimensions
    num_indices, num_cols = grad_output.shape
    assert index.shape[0] == num_indices, "Index length must match the number of rows in grad_output"
    assert grad_source.shape[1] == num_cols, "Number of columns in grad_source must match grad_output"

    # Define block sizes
    BLOCK_SIZE_INDEX = 128
    BLOCK_SIZE_COL = 128

    # Calculate grid size
    grid = (
        (num_indices + BLOCK_SIZE_INDEX - 1) // BLOCK_SIZE_INDEX,
        (num_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    )

    # Launch the Triton kernel
    index_select_cat_bwd_kernel[grid](
        grad_output_ptr=grad_output,
        index_ptr=index,
        grad_source_ptr=grad_source,
        num_indices=num_indices,
        num_cols=num_cols,
        BLOCK_SIZE_INDEX=BLOCK_SIZE_INDEX,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

    return grad_source
