import triton
import triton.language as tl
import torch

# Triton kernel for computing the sine of elements in a tensor
@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate the start of the block and the offsets within the block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load the input data with masking
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute the sine of the input data
    output = tl.sin(x)
    
    # Store the result back to the output pointer with masking
    tl.store(out_ptr + offsets, output, mask=mask)

# Function to call the Triton kernel
def sin_triton(x, out):
    # Determine the number of elements in the input tensor
    n_elements = x.numel()
    
    # Launch the Triton kernel
    sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

# Example usage
if __name__ == "__main__":
    # Create input and output tensors
    x = torch.randn(1024, device='cuda')
    out = torch.empty_like(x)
    
    # Call the wrapper function to compute the sine
    sin_triton(x, out)
    
    # Verify the result
    assert torch.allclose(out, torch.sin(x))
    print("Computation is correct!")
