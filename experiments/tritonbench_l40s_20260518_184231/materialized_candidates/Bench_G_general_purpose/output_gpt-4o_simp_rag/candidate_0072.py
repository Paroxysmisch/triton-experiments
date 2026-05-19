import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(
    input_ptr, output_ptr, 
    BLOCK_SIZE: tl.constexpr, 
    num_cols: tl.constexpr
):
    # Get the program ID, which represents the row being processed
    pid = tl.program_id(0)
    
    # Calculate the start index for the row
    row_start = pid * num_cols
    
    # Load the input row into registers
    input_row = tl.load(input_ptr + row_start + tl.arange(0, num_cols))
    
    # Compute the square of each element
    output_row = input_row * input_row
    
    # Store the result back to the output pointer
    tl.store(output_ptr + row_start + tl.arange(0, num_cols), output_row)

def square(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the input is a 2D tensor
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    
    # Determine the shape of the input tensor
    num_rows, num_cols = input_tensor.shape
    
    # Create an output tensor with the same shape as the input
    output_tensor = torch.empty_like(input_tensor)
    
    # Define the block size
    BLOCK_SIZE = 128  # or any appropriate size based on hardware and input size
    
    # Launch the kernel for each row
    grid = (num_rows,)  # Launch one block per row
    square_kernel[grid](
        input_tensor, output_tensor, 
        BLOCK_SIZE=BLOCK_SIZE, 
        num_cols=num_cols
    )
    
    return output_tensor
