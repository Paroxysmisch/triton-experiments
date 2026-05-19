import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    row_stride,  # Stride between rows
    n_cols,      # Number of columns
    mask_ptr,    # Optional attention mask pointer
    BLOCK_SIZE: tl.constexpr,  # Size of the block to process
):
    # Get the row index using program ID
    row_idx = tl.program_id(0)
    
    # Compute pointers for the current row
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # Create offsets for the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load input row with mask for out-of-bounds accesses
    row_mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=row_mask, other=-float('inf'))
    
    # Apply attention mask if provided
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * n_cols + col_offsets
        mask = tl.load(mask_ptrs, mask=row_mask, other=-float('inf'))
        row = row + mask
    
    # Compute softmax
    row_max = tl.max(row, axis=0)
    row = row - row_max  # Subtract max for numerical stability
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Write output
    output_row_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=row_mask)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    # Handle input validation and reshaping
    if dim != -1:
        input = input.transpose(dim, -1)
    
    orig_shape = input.shape
    if input.dim() > 2:
        input = input.reshape(-1, input.size(-1))
    
    # Get tensor dimensions
    n_rows, n_cols = input.shape
    
    # Compute block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Determine number of warps based on problem size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    
    # Launch kernel
    grid = lambda meta: (n_rows,)
    softmax_kernel[grid](
        output,
        input,
        input.stride(0),
        n_cols,
        mask if mask is not None else None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Restore original shape if needed
    if input.dim() != len(orig_shape):
        output = output.reshape(orig_shape)
    if dim != -1:
        output = output.transpose(dim, -1)
    
    return output
