import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, row_stride, n_cols,
    mask_ptr, BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Compute pointers for this row
    row_start_ptr = input_ptr + row_idx * row_stride
    
    # Create offsets for this BLOCK
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create block mask for handling incomplete blocks
    block_mask = col_offsets < n_cols
    
    # Load input row into SRAM
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=block_mask, other=-float('inf'))
    
    # Apply mask if provided
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask = tl.load(mask_ptrs, mask=block_mask, other=0)
        row = row + mask * -float('inf')
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max from row and compute exponentials
    row = row - row_max
    numerator = tl.exp(row)
    
    # Compute sum for denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Write output
    output_ptrs = output_ptr + row_idx * row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=block_mask)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    # Handle dimension and shape validation
    if dim != -1 and dim != input.dim() - 1:
        input = input.transpose(dim, -1)
    
    # Reshape input into 2D tensor
    orig_shape = input.shape
    if input.dim() > 2:
        input = input.reshape(-1, input.size(-1))
    
    # Get dimensions
    n_rows, n_cols = input.shape
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Handle mask
    if mask is not None:
        if mask.shape != input.shape:
            raise ValueError("Mask shape must match input shape")
        mask = mask.reshape(-1, mask.size(-1))
    
    # Calculate optimal block size
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Maximum block size
    
    # Calculate grid and num_warps based on problem size
    if n_rows > 8192:
        grid = lambda meta: (n_rows,)
        num_warps = 4
    else:
        grid = (n_rows,)
        num_warps = 1
    
    # Launch kernel
    softmax_kernel[grid](
        output.data_ptr(),
        input.data_ptr(),
        input.stride(-2),
        n_cols,
        mask.data_ptr() if mask is not None else None,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Reshape output back to original shape if needed
    if len(orig_shape) > 2:
        output = output.reshape(orig_shape)
    
    return output
