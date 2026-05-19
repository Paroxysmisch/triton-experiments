import torch
import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input vector
    y_ptr,  # Pointer to the second input vector
    output_ptr,  # Pointer to the output vector
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr  # Size of each block
):
    # Get the program ID for this block
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load x and y from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the addition
    result = x + y
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)

# Define the wrapper function
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Get the number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    add_kernel[grid](
        x_ptr=x, 
        y_ptr=y, 
        output_ptr=output, 
        n_elements=n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create CUDA tensors
    x = torch.randn(10240, device='cuda')
    y = torch.randn(10240, device='cuda')
    
    # Call the add function
    result = add(x, y)
    
    # Print the result
    print(result)
