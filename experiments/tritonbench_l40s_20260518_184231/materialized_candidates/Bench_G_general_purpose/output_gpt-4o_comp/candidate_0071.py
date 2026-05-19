import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    input_ptr, output_ptr,
    input_stride, output_stride,
    n_cols, BLOCK_SIZE: tl.constexpr
):
    # Program ID corresponds to the row index
    row_idx = tl.program_id(0)
    
    # Calculate the start of the row for input and output
    input_offset = row_idx * input_stride
    output_offset = row_idx * output_stride
    
    # Create a block of indices for the row
    col_indices = tl.arange(0, BLOCK_SIZE)
    
    # Load the row into SRAM with masking
    mask = col_indices < n_cols
    row = tl.load(input_ptr + input_offset + col_indices, mask=mask, other=0.0)
    
    # Square each element
    square_output = row * row
    
    # Store the squared row back to global memory
    tl.store(output_ptr + output_offset + col_indices, square_output, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    # Get the shape of the input tensor
    n_rows, n_cols = x.shape
    
    # Determine BLOCK_SIZE as the smallest power of two greater than n_cols
    BLOCK_SIZE = 2 ** (n_cols - 1).bit_length()
    
    # Determine the number of warps to use based on BLOCK_SIZE
    num_warps = min(4, (BLOCK_SIZE + 31) // 32)  # Choose 1, 2, or 4 warps
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = (n_rows,)  # 1D grid with one instance per row
    square_kernel[grid](
        x.data_ptr(), y.data_ptr(),
        x.stride(0), y.stride(0),
        n_cols, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return y

# Example usage:
# x = torch.rand((1024, 512), device='cuda')
# y = square(x)
