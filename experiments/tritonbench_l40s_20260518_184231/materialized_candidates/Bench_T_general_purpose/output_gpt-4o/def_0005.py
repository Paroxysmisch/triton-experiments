import triton
import triton.language as tl
import torch

@triton.jit
def relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the block ID and offset
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Apply ReLU and then square root
    relu_data = tl.max(input_data, 0.0)
    sqrt_data = tl.sqrt(relu_data)

    # Store the result
    tl.store(output_ptr + offsets, sqrt_data, mask=offsets < n_elements)

def relu_sqrt(input, inplace=False, out=None):
    # Ensure input is a contiguous tensor
    input = input.contiguous()

    # Determine the number of elements
    n_elements = input.numel()

    # Handle in-place operation
    if inplace:
        output = input
    else:
        # Allocate output tensor if not provided
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # Define a suitable block size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_sqrt_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output

# Example usage
input_tensor = torch.tensor([-1.0, 0.0, 1.0, 4.0, 9.0], dtype=torch.float32)
output_tensor = relu_sqrt(input_tensor)
print(output_tensor)
