import triton
from triton.utils import nvcc_required

@nvcc_required
def selu(input, inplace=False):
    # Check if the input is a Triton Tensor
    if not isinstance(input, triton.Tensor):
        raise ValueError("Input must be a Triton Tensor")

    # Get the device and data type of the input
    device = input.device
    dtype = input.dtype

    # Create a new output tensor if inplace is False
    if not inplace:
        output = triton.empty_like(input)
    else:
        output = input

    # Define the grid size and block size
    N_elements = input.numel()
    BLOCK_SIZE = 256
    grid_size = (N_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    selu_kernel[grid_size, BLOCK_SIZE](input.data_ptr(), output.data_ptr(), N_elements)

    return output
