import triton
import triton.language as tl

# Define the GELU kernel
@triton.jit
def gelu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    approximate,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    # Approximate GELU using tanh
    if approximate == 'tanh':
        cdf_x = 0.5 * x * (1 + tl.tanh(tl.sqrt(2 / tl.float32(math.pi)) * (x + 0.044715 * x ** 3)))
    else:
        # Exact GELU calculation
        cdf_x = x * tl.cdf_normal(x)

    tl.store(output_ptr + offsets, cdf_x, mask=mask)

# Define the wrapper function
def gelu(input, approximate='none'):
    # Get the shape and dtype of the input tensor
    input_shape = input.shape
    input_dtype = input.dtype

    # Create an output tensor with the same shape and dtype
    output = tl.zeros_like(input)

    # Get the number of elements in the input tensor
    n_elements = input.size

    # Launch the kernel
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    gelu_kernel[grid_size, BLOCK_SIZE](output.data, input.data, n_elements, approximate)

    return output

# Example usage
import numpy as np
input_tensor = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
result = gelu(input_tensor, approximate='tanh')
print(result)
