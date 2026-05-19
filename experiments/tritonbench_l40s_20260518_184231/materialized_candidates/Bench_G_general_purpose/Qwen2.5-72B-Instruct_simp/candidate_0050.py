import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,  # Pointer to the input tensor
    out_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size (number of elements processed per block)
):
    # Compute the block ID and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a vectorized index for the elements this block will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Apply the ReLU operation
    out = tl.where(x > 0, x, 0)

    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, out, mask=mask)

import torch
import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE = 1024

# Python wrapper function
def relu(x: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"

    # Allocate the output tensor
    out = torch.empty_like(x)

    # Compute the number of elements
    n_elements = x.numel()

    # Compute the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    relu_kernel[grid, BLOCK_SIZE](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    x = torch.randn(1024 * 1024, device="cuda")

    # Apply the ReLU operation
    out = relu(x)

    # Print the first few elements of the output
    print(out[:10])
