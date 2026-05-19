import triton
import triton.language as tl
import torch

# Define the Triton kernel
@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for each element in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only process valid elements
    mask = offsets < n_elements
    
    # Load data from the input pointer, applying the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute the sine of the input data
    y = tl.math.sin(x)
    
    # Store the result back to the output pointer
    tl.store(output_ptr + offsets, y, mask=mask)

# Define the wrapper function to call the kernel
def call_kernel(x, BLOCK_SIZE=1024):
    # Ensure input is a torch tensor
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    # Flatten the input tensor to ensure it's 1D
    x = x.flatten()
    
    # Calculate the total number of elements
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define grid configuration
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.rand(10240, device='cuda')
    
    # Call the kernel
    output = call_kernel(x)
    
    # Print the result
    print(output)
