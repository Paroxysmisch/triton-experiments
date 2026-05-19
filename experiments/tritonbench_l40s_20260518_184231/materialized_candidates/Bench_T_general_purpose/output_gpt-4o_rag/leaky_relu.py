import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x_ptr, y_ptr, negative_slope, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define the program ID and offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a range of offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the input data
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute the Leaky ReLU
    zero = 0.0
    y = tl.where(x >= zero, x, negative_slope * x)

    # Store the result
    tl.store(y_ptr + offsets, y, mask=offsets < n_elements)

import torch

def leaky_relu(input, negative_slope=0.01, inplace=False):
    # Ensure input is a contiguous tensor
    input = input.contiguous()

    # Allocate output tensor
    output = input if inplace else torch.empty_like(input)

    # Get the number of elements
    n_elements = input.numel()

    # Define the block size
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU's capabilities

    # Launch the Triton kernel
    leaky_relu_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE](
        input.data_ptr(),
        output.data_ptr(),
        negative_slope,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output
