import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr,
    output_ptr,
    index_ptr,
    input_row_stride,
    output_row_stride,
    index_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=float('inf'))
    
    min_val = tl.min(row, axis=0)
    min_index = tl.argmin(row, axis=0)
    
    output_row = output_ptr + row_idx * output_row_stride
    index_row = index_ptr + row_idx * index_row_stride
    tl.store(output_row, min_val)
    tl.store(index_row, min_index)

def min(input, dim, keepdim=False, *, out=None):
    assert dim >= 0 and dim < input.dim(), f"dim {dim} is out of bounds"
    original_shape = input.shape
    
    # Transpose the target dimension to the last position
    input_transposed = input.transpose(dim, -1)
    transposed_shape = input_transposed.shape
    
    # Flatten to 2D tensor (n_rows, n_cols)
    input_2d = input_transposed.flatten(end_dim=-2)
    n_rows, n_cols = input_2d.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensors
    min_values = torch.empty((n_rows, 1), device=input.device, dtype=input.dtype)
    min_indices = torch.empty((n_rows, 1), device=input.device, dtype=torch.long)
    
    # Launch kernel
    grid = (n_rows,)
    min_kernel[grid](
        input_2d,
        min_values,
        min_indices,
        input_2d.stride(0),
        min_values.stride(0),
        min_indices.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to transposed shape with last dimension reduced to 1
    min_values = min_values.view(*transposed_shape[:-1], 1)
    min_indices = min_indices.view(*transposed_shape[:-1], 1)
    
    # Transpose back if keepdim is True, else squeeze
    if keepdim:
        min_values = min_values.transpose(dim, -1)
        min_indices = min_indices.transpose(dim, -1)
    else:
        min_values = min_values.squeeze(-1)
        min_indices = min_indices.squeeze(-1)
    
    # Handle output tuple
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a tuple of two tensors")
        out[0].copy_(min_values)
        out[1].copy_(min_indices)
    
    return (min_values, min_indices)
