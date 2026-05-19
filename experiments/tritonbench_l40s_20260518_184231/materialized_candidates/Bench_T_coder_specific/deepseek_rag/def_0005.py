import triton.language as tl
import torch

@triton.jit
def relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor with boundary mask
    input_value = tl.load(input_ptr + offset, mask=mask)
    # Apply ReLU function
    input_value = tl.max(input_value, 0.0)
    # Compute the square root of the loaded elements
    output_value = tl.sqrt(input_value.to(tl.float32))
    # Store the result in output tensor with boundary mask
    tl.store(output_ptr + offset, output_value, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if out is None:
        if inplace:
            out = input
        else:
            out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        if inplace:
            assert out.device == input.device, "Inplace operation requires input and output to be on the same device"
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    if inplace:
        relu_sqrt_kernel[(grid_size, )](input.data_ptr(), input.data_ptr(), n_elements, BLOCK_SIZE)
    else:
        relu_sqrt_kernel[(grid_size, )](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE)
    return out
