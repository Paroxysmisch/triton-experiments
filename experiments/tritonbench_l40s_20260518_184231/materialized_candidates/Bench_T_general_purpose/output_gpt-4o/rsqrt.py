import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program's start index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a block of indices
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input values
    input_values = tl.load(input_ptr + offsets, mask=mask)

    # Compute the reciprocal of the square root
    result = 1.0 / tl.sqrt(input_values)

    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

# Wrapper function
def rsqrt(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements
    n_elements = input.numel()

    # Define block size
    BLOCK_SIZE = 1024  # Example block size

    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    rsqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
input_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float32)
output_tensor = rsqrt(input_tensor)
print(output_tensor)  # Should print tensor([1.0000, 0.5000, 0.3333, 0.2500])
