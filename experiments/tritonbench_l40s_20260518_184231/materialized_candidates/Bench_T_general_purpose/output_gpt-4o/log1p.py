import triton
import triton.language as tl
import torch

@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Create a block of pointers to the input and output data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load the input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the natural logarithm of (1 + x)
    y = tl.log(x + 1.0)
    
    # Store the result
    tl.store(output_ptr + offsets, y, mask=mask)

def log1p(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024  # You can adjust this based on your hardware capabilities
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage:
input_tensor = torch.tensor([0.1, 0.01, 0.001], dtype=torch.float32)
output_tensor = log1p(input_tensor)
print(output_tensor)
