import triton
import triton.language as tl
import torch

# Kernel function to apply ReLU and then square root to each element in the tensor
@triton.jit
def relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load elements from the input tensor
    input_values = tl.load(input_ptr + offsets, mask=mask)

    # Apply ReLU and then square root
    output_values = tl.sqrt(tl.max(input_values, 0.0))

    # Store the result in the output tensor
    tl.store(output_ptr + offsets, output_values, mask=mask)

# Wrapper function to launch the Triton kernel and apply ReLU and square root
def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    if inplace and out is not None:
        raise ValueError("Cannot specify both 'inplace=True' and 'out' at the same time.")
    
    if inplace:
        out = input
    elif out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = 1024  # Adjust block size as needed
    grid_size = (n_elements + block_size - 1) // block_size

    # Launch the Triton kernel
    relu_sqrt_kernel[(grid_size,)](input, out, n_elements, BLOCK_SIZE=block_size)

    return out
