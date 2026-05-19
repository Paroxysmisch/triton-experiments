import triton
import triton.language as tl

# Define the block size for processing elements
BLOCK_SIZE = 128

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index and the range of elements this block will handle
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform element-wise addition
    result = x + y

    # Store the result in the output tensor
    tl.store(output_ptr + offsets, result, mask=mask)

def add_wrapper(x, y, output, n_elements):
    # Calculate the number of blocks needed
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
if __name__ == "__main__":
    import torch

    # Initialize input tensors
    n_elements = 1024
    x = torch.rand(n_elements, dtype=torch.float32).cuda()
    y = torch.rand(n_elements, dtype=torch.float32).cuda()
    output = torch.empty_like(x)

    # Call the wrapper function
    add_wrapper(x, y, output, n_elements)

    # Verify the result
    assert torch.allclose(output, x + y)
