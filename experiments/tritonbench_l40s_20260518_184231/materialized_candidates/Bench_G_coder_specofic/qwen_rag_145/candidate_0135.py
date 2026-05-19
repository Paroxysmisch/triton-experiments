import triton.language as tl
import triton

BLOCK_SIZE = 256

@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculating starting point for each program instance
    block_start = tl.program_id(0) * BLOCK_SIZE

    # Creating offsets tensor for element indexing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask ensures operations only occur on valid indices within the input bounds
    mask = offsets < n_elements

    # Loading data from x_ptr
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute sine value using tl.math.sin
    y = tl.math.sin(x)

    # Store result in output_ptr
    tl.store(output_ptr + offsets, y, mask=mask)

def call_kernel(x_ptr, output_ptr, n_elements):
    # Calculating total number of elements
    n_elements = n_elements

    # Initializing output tensor
    output_tensor = triton.language.Tensor([n_elements])

    # Configuring the grid
    grid = lambda metadata : (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launching the kernel
    kernel_function[(grid,)](x_ptr, output_ptr, n_elements, BLOCK_SIZE)
