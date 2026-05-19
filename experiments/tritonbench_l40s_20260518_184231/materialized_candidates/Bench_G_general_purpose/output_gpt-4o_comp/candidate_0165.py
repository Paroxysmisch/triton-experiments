import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    # Compute the row index
    row_idx = tl.program_id(0)
    # Compute the starting pointer for the current row
    row_start = row_idx * row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute input pointers for the current row
    input_ptrs = input_ptr + row_start + col_offsets
    
    # Load the input row into SRAM
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Compute the maximum value for numerical stability
    row_max = tl.max(row, axis=0)
    row = row - row_max
    
    # Apply mask if provided
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + row_start + col_offsets, mask=col_offsets < n_cols, other=0)
        row = row + mask
    
    # Compute exponentials
    exp_row = tl.exp(row)
    
    # Compute the sum of exponentials
    denom = tl.sum(exp_row, axis=0)
    
    # Compute the softmax
    softmax_row = exp_row / denom
    
    # Store the result back to the output tensor
    output_ptrs = output_ptr + row_start + col_offsets
    tl.store(output_ptrs, softmax_row, mask=col_offsets < n_cols)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    assert dim == -1, "This implementation only supports softmax along the last dimension."
    
    # Reshape input tensor to 2D if necessary
    if input.dim() != 2:
        input = input.view(-1, input.size(-1))
    
    # Determine the size of the input
    n_rows, n_cols = input.shape
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Determine block size and number of warps
    BLOCK_SIZE = 128
    num_warps = 4 if n_cols > 128 else 1
    
    # Determine grid size
    grid = lambda meta: (n_rows,)
    
    # Launch the Triton kernel
    softmax_kernel[grid](
        output,
        input,
        input.stride(0),
        n_cols,
        mask,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output
