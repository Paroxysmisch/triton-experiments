import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for parallel execution
    pid = tl.program_id(0)
    
    # Create a block of indices for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input elements
    x = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Compute the sigmoid function
    result = 1 / (1 + tl.exp(-x))
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=offsets < n_elements)


import torch

def sigmoid(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Flatten the input tensor to handle it as a 1D array
    input_flat = input.flatten()
    n_elements = input_flat.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input_flat)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("Output must be a torch.Tensor")
        if out.numel() != n_elements:
            raise ValueError("Output tensor must have the same number of elements as input")

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Choose an appropriate block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    sigmoid_kernel[grid](
        input_flat.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape the output tensor to match the input shape
    return out.view_as(input)

# Example usage
input_tensor = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float32)
output_tensor = sigmoid(input_tensor)
print(output_tensor)
