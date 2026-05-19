import triton as tl
import numpy as np

BLOCK_SIZE = 1024

@tl.jit
def kernel_function(x_ptr, output_ptr, n_elements):
    # Get the program instance's index and calculate the starting point
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create an offsets tensor for element indexing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask ensures operations only occur on valid indices within the input bounds
    mask = offsets < n_elements

    # Load data from x_ptr, compute the sine using tl.math.sin, and store the result in output_ptr
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.math.sin(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def call_kernel(x_ptr, output_ptr, n_elements):
    # Calculate the total number of elements
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create an output tensor
    output = tl.empty(shape=(n_elements,), dtype=tl.float32)

    # Define a grid configuration function using lambda to handle thread block calculations
    grid = lambda meta: (n_blocks, 1, 1)

    # Launch the kernel
    kernel_function[grid](x_ptr, output_ptr, n_elements)

    return output
