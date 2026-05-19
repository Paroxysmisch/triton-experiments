import triton
import triton.language as tl
import torch

# Triton kernel for computing the sine of elements in a tensor
@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    # Get the program ID (which block we are processing)
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create an array of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only access valid elements
    mask = offsets < n_elements
    
    # Load the input data using the mask to prevent out-of-bounds access
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute the sine of each element
    output = tl.sin(x)
    
    # Store the result back into memory, using the mask to ensure only valid memory locations are written
    tl.store(out_ptr + offsets, output, mask=mask)

# Function to call the Triton kernel
def sin_triton(x, out):
    # Determine the number of elements in the input tensor
    n_elements = x.numel()
    
    # Invoke the Triton kernel with the appropriate grid size and block size
    sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1024, device='cuda')
    
    # Create an output tensor with the same shape and type as the input tensor
    out = torch.empty_like(x)
    
    # Call the sin_triton function to compute the sine of the input tensor
    sin_triton(x, out)
    
    # Print the result
    print(out)
