import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(input_ptr, output_min_ptr, output_idx_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate the start of the row in input and output
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_min_row_start_ptr = output_min_ptr + row_idx * output_row_stride
    output_idx_row_start_ptr = output_idx_ptr + row_idx * output_row_stride
    
    # Initialize the minimum value and index
    min_val = tl.load(row_start_ptr)
    min_idx = tl.zeros([1], dtype=tl.int32)
    
    # Iterate over the row in BLOCK_SIZE chunks
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets + col_offset
        mask = col_offsets + col_offset < n_cols
        
        # Load the current block of the row
        values = tl.load(input_ptrs, mask=mask, other=float('inf'))
        
        # Find the minimum value and index in the current block
        min_mask = values < min_val
        min_val = tl.where(min_mask, values, min_val)
        min_idx = tl.where(min_mask, col_offsets + col_offset, min_idx)
    
    # Store the results
    tl.store(output_min_row_start_ptr, min_val)
    tl.store(output_idx_row_start_ptr, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    assert dim == 1, "This implementation currently supports reduction along dimension 1 only."
    
    # Reshape input if needed
    input_reshaped = input.reshape(-1, input.shape[-1])
    n_rows, n_cols = input_reshaped.shape
    
    # Prepare output tensors
    if out is None:
        min_values = torch.empty((n_rows, 1) if keepdim else (n_rows,), device=input.device, dtype=input.dtype)
        min_indices = torch.empty((n_rows, 1) if keepdim else (n_rows,), device=input.device, dtype=torch.int32)
    else:
        min_values, min_indices = out
    
    # Define block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Launch kernel
    min_kernel[(n_rows,)](
        input_reshaped, min_values, min_indices,
        input_reshaped.stride(0), min_values.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape outputs if keepdim is False
    if not keepdim:
        min_values = min_values.squeeze(dim)
        min_indices = min_indices.squeeze(dim)
    
    return min_values, min_indices

# Example usage
x = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]], device='cuda')
min_vals, min_idxs = min(x, dim=1, keepdim=True)
print(min_vals)  # Output: tensor([[1.], [4.]], device='cuda:0')
print(min_idxs)  # Output: tensor([[1], [1]], device='cuda:0')
