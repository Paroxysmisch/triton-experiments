import triton
import triton.language as tl

@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the block and thread index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't read/write out of bounds
    mask = offsets < n_elements

    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)

    # Compute tanh
    output_data = tl.libdevice.tanh(input_data)

    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)

def tanh(input, *, out=None):
    # Ensure the input is a Triton tensor
    assert isinstance(input, triton.Tensor), "Input must be a Triton tensor"

    # If out is not provided, create an output tensor with the same shape as input
    if out is None:
        out = triton.zeros_like(input)

    # Ensure the output is a Triton tensor
    assert isinstance(out, triton.Tensor), "Output must be a Triton tensor"

    # Get the number of elements in the input tensor
    n_elements = input.numel()

    # Define block size (you may need to tune this for performance)
    BLOCK_SIZE = 1024

    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out
