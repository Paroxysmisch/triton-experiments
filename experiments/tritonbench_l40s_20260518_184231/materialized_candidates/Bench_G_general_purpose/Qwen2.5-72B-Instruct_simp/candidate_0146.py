import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,  # Pointer to input tensor x
    y_ptr,  # Pointer to input tensor y
    z_ptr,  # Pointer to input tensor z
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensors
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Calculate the start index for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Create offsets for the block
    mask = offsets < n_elements  # Create a mask to handle the case where the block extends beyond the tensor

    x = tl.load(x_ptr + offsets, mask=mask)  # Load the elements from x
    y = tl.load(y_ptr + offsets, mask=mask)  # Load the elements from y
    z = tl.load(z_ptr + offsets, mask=mask)  # Load the elements from z

    # Perform the fused operation: (x + y) * z
    result = (x + y) * z

    # Apply an activation function (e.g., ReLU)
    result = tl.where(result > 0, result, 0)

    tl.store(output_ptr + offsets, result, mask=mask)  # Store the result back to the output tensor

import torch
import triton
import triton.language as tl

def fused_add_mul_activation_torch(x, y, z):
    # Ensure the input tensors are on the same device and have the same shape
    assert x.device == y.device == z.device, "Tensors must be on the same device"
    assert x.shape == y.shape == z.shape, "Tensors must have the same shape"

    # Get the number of elements in the tensors
    n_elements = x.numel()

    # Allocate the output tensor
    output = torch.empty_like(x)

    # Define the grid and block dimensions
    grid = (triton.cdiv(n_elements, 1024),)
    block = (1024,)

    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid, block](
        x, y, z, output, n_elements, BLOCK_SIZE=1024
    )

    return output
