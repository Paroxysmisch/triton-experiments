import torch
import triton
import triton.language as tl

# Define the Triton kernel for element-wise addition
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Program ID in the block dimension
    pid = tl.program_id(0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Apply a mask to ensure we do not access out-of-bound elements
    mask = offsets < n_elements
    
    # Load input values with the mask applied
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    
    # Perform the element-wise addition
    result = x + y
    
    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, result, mask=mask)

# Define the wrapper function to set up and execute the kernel
def add_wrapper(x, y, BLOCK_SIZE=1024):
    # Initialize the output tensor with zeros, same shape and dtype as x
    out = torch.zeros_like(x)
    
    # Get the total number of elements
    n_elements = x.numel()
    
    # Calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    
    # Return the result
    return out

# Example usage
if __name__ == "__main__":
    # Initialize two input tensors
    x = torch.rand(1024, device='cuda')
    y = torch.rand(1024, device='cuda')
    
    # Call the wrapper function to perform addition
    result = add_wrapper(x, y)
    
    # Print the result
    print(result)
