import triton
import triton.language as tl
import torch

@triton.jit
def sqrt_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program index
    pid = tl.program_id(axis=0)
    
    # Compute the start and end index for this block
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, n_elements)
    
    # Create a range for the block
    offsets = start + tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute sqrt and then tanh
    sqrt_data = tl.sqrt(input_data)
    result = tl.tanh(sqrt_data)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)

def sqrt_tanh(input, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure the output tensor is of the same shape as input
    if out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU's capabilities
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage:
input_tensor = torch.tensor([1.0, 4.0, -1.0, 9.0], device='cuda')
output_tensor = sqrt_tanh(input_tensor)
print(output_tensor)
