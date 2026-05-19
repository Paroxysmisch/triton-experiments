import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    values_ptr,
    indices_ptr,
    input_ptr,
    input_row_stride,
    values_row_stride,
    indices_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    max_value = tl.max(row, axis=0)
    max_index = tl.argmax(row, axis=0)
    
    values_row_ptr = values_ptr + row_idx * values_row_stride
    indices_row_ptr = indices_ptr + row_idx * indices_row_stride
    tl.store(values_row_ptr, max_value)
    tl.store(indices_row_ptr, max_index)

def max(input, dim, keepdim=False, *, out=None):
    assert input.dim() > 0, "input must be at least 1D"
    assert dim >= 0 and dim < input.dim(), f"dim {dim} is out of range [0, {input.dim()-1}]"
    
    # Move the reduced dimension to the end and make contiguous
    input_ = input.transpose(dim, -1).contiguous()
    dim_size = input_.shape[-1]
    
    # Flatten all dimensions except the last (reduced) into a single dimension
    input_flat = input_.view(-1, dim_size)
    num_rows, n_cols = input_flat.shape
    
    # Determine block size for the kernel
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensors
    values_shape = (num_rows, 1) if keepdim else (num_rows,)
    values = torch.empty(values_shape, dtype=input.dtype, device=input.device)
    indices = torch.empty(values_shape, dtype=torch.long, device=input.device)
    
    # Launch kernel
    grid = (num_rows,)
    max_kernel[grid](
        values, indices,
        input_flat,
        input_flat.stride(0),
        values.stride(0),
        indices.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape and transpose back if needed
    if keepdim:
        output_shape = list(input_.shape[:-1]) + [1]
        values = values.view(output_shape)
        indices = indices.view(output_shape)
        values = values.transpose(-1, dim)
        indices = indices.transpose(-1, dim)
    else:
        output_shape = list(input_.shape[:-1])
        values = values.view(output_shape)
        indices = indices.view(output_shape)
    
    # Handle out parameter
    if out is not None:
        out_values, out_indices = out
        out_values.copy_(values)
        out_indices.copy_(indices)
        return out
    else:
        return (values, indices)
