import triton
import triton.language as tl
import torch

# Triton kernel to compute the square root element-wise
@triton.jit
def sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the unique program index
    pid = tl.program_id(0)
    
    # Create a block of indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load the input values
    input_vals = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute the square root
    output_vals = tl.sqrt(input_vals)
    
    # Store the results
    tl.store(output_ptr + offsets, output_vals, mask=offsets < n_elements)

# Wrapper function
def sqrt(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Ensure the output is a torch tensor
    if not isinstance(out, torch.Tensor):
        raise TypeError("Output must be a torch.Tensor")

    # Flatten the input and output tensors
    input_flat = input.flatten()
    out_flat = out.flatten()

    # Number of elements
    n_elements = input_flat.numel()

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a suitable block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    sqrt_kernel[grid](input_flat, out_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape the output to match the input shape
    return out_flat.view_as(input)

# Example usage
input_tensor = torch.tensor([1.0, 4.0, 9.0, -1.0], dtype=torch.float32)
output_tensor = sqrt(input_tensor)
print(output_tensor)  # Expected output: tensor([1.0, 2.0, 3.0, nan])
