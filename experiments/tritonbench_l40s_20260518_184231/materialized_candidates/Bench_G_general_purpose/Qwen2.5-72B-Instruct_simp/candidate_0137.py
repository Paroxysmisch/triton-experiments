import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input tensor
    y_ptr,  # Pointer to the second input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Compute the starting index for the current block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the elements from the input tensors
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements)

    # Perform the element-wise addition
    output = x + y

    # Store the result in the output tensor
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

import torch

def add_wrapper(x, y, output, block_size):
    # Ensure the input tensors are on the same device
    assert x.device == y.device, "Input tensors must be on the same device"
    assert x.device == output.device, "Output tensor must be on the same device"

    # Get the number of elements
    n_elements = x.numel()

    # Define the grid and block dimensions
    grid = (triton.cdiv(n_elements, block_size),)

    # Launch the kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)

# Example usage
if __name__ == "__main__":
    # Define the block size
    block_size = 128

    # Create input tensors
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')

    # Initialize the output tensor
    output = torch.empty_like(x)

    # Call the wrapper function
    add_wrapper(x, y, output, block_size)

    # Verify the result
    expected_output = x + y
    assert torch.allclose(output, expected_output), "The output is incorrect"
    print("Output is correct!")
