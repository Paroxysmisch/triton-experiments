import triton
import triton.language as tl
import torch

# Triton kernel to compute the absolute value of each element
@triton.jit
def abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the block index and the offset within the block
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load the input values
    input_vals = tl.load(input_ptr + offset, mask=offset < n_elements, other=0.0)
    
    # Compute the absolute values
    abs_vals = tl.abs(input_vals)
    
    # Store the results in the output
    tl.store(output_ptr + offset, abs_vals, mask=offset < n_elements)

# Wrapper function
def abs(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # If out is not provided, create an output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        # Ensure the output tensor is the same size as the input
        if out.size() != input.size():
            raise ValueError("out tensor must have the same size as the input tensor")
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Define the block size (tune this for optimal performance)
    BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    abs_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage
input_tensor = torch.tensor([-1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
output_tensor = abs(input_tensor)
print(output_tensor)  # Expected output: tensor([1.0, 2.0, 3.0, 4.0])
