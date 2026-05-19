import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    # Calculate the offset and index for each block
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    
    # Ensure we do not access out-of-bounds elements
    xmask = xindex < xnumel
    
    # Load the input data
    x = tl.load(in_ptr + xindex, mask=xmask, other=0.0)
    
    # Compute the sine of the input data
    result = tl.sin(x)
    
    # Store the result in the output pointer
    tl.store(out_ptr + xindex, result, mask=xmask)

def call_kernel(x):
    # Ensure input is a torch tensor
    assert isinstance(x, torch.Tensor)
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Number of elements in the input tensor
    xnumel = x.numel()
    
    # Define block size for parallel execution
    XBLOCK = 1024  # You can adjust this based on your GPU's capabilities
    
    # Launch the Triton kernel
    grid = (xnumel + XBLOCK - 1) // XBLOCK  # Calculate number of blocks
    kernel_function[grid](x, output, xnumel, XBLOCK=XBLOCK)
    
    return output

# Example usage:
x = torch.rand(1024, device='cuda')  # Create a random tensor on GPU
result = call_kernel(x)  # Compute the sine of each element
print(result)  # Output the result
