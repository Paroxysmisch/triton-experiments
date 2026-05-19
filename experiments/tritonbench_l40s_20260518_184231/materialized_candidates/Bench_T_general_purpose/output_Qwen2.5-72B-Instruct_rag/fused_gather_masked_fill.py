import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    output_ptr, input_ptr, index_ptr, mask_ptr, value, 
    input_stride, index_stride, mask_stride, 
    n_elements, n_cols, BLOCK_SIZE: tl.constexpr
):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate the starting pointers for the current row
    input_row_start_ptr = input_ptr + row_idx * input_stride
    index_row_start_ptr = index_ptr + row_idx * index_stride
    mask_row_start_ptr = mask_ptr + row_idx * mask_stride
    
    # Load the indices for the current row
    indices = tl.load(index_row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=0)
    
    # Load the input values for the current row
    input_values = tl.load(input_row_start_ptr + indices * n_elements, mask=col_offsets < n_cols, other=0)
    
    # Load the mask for the current row
    mask_values = tl.load(mask_row_start_ptr + col_offsets, mask=col_offsets < n_cols, other=0)
    
    # Apply the mask to replace the gathered elements with the specified value
    output_values = tl.where(mask_values, value, input_values)
    
    # Write back the output to DRAM
    output_row_start_ptr = output_ptr + row_idx * input_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output_values, mask=col_offsets < n_cols)

import torch

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Validate input tensors and their shapes
    if input.dim() != index.dim():
        raise ValueError("Input and index tensors must have the same number of dimensions.")
    if any(input.size(d) != index.size(d) for d in range(input.dim()) if d != dim):
        raise ValueError("The size of index at each dimension d must not exceed the size of input at that dimension, except at dimension dim.")
    if not mask.shape == input.shape:
        raise ValueError("The mask tensor must be broadcastable to the shape of the output tensor.")
    
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Determine the grid and block sizes
    n_elements = input.size(dim)
    n_cols = input.size(0) if dim == 0 else input.size(1)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Launch the Triton kernel
    fused_gather_masked_fill_kernel[(n_elements,)](
        out, input, index, mask, value,
        input.stride(dim), index.stride(dim), mask.stride(dim),
        n_elements, n_cols, BLOCK_SIZE
    )
    
    return out
