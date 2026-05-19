import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    output_ptr,  # Pointer to the output tensor
    input_ptr,   # Pointer to the input tensor
    input_shape, # Shape of the input tensor
    input_strides, # Strides of the input tensor
    dim,         # Dimension to reduce
    keepdim,     # Whether to keep the reduced dimension
    n_elements,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Initialize the maximum value and its index
    max_val = -float('inf')
    max_idx = -1

    for i in range(block_start, min(block_start + BLOCK_SIZE, n_elements)):
        # Compute the linear index in the input tensor
        linear_idx = i
        # Compute the coordinates in the input tensor
        coords = [0] * len(input_shape)
        for d in range(len(input_shape) - 1, -1, -1):
            coords[d] = linear_idx % input_shape[d]
            linear_idx //= input_shape[d]

        # Compute the index in the input tensor
        input_idx = sum(c * s for c, s in zip(coords, input_strides))

        # Load the value from the input tensor
        value = tl.load(input_ptr + input_idx)

        # Update the maximum value and its index
        if value > max_val:
            max_val = value
            max_idx = coords[dim]

    # Write the maximum index to the output tensor
    if max_idx != -1:
        output_idx = pid
        if keepdim:
            output_idx = output_idx * input_shape[dim] + max_idx
        tl.store(output_ptr + output_idx, max_idx)

import torch

def argmax(input, dim=None, keepdim=False):
    # Flatten the input tensor if dim is None
    if dim is None:
        input = input.flatten()
        dim = 0

    # Get the shape and strides of the input tensor
    input_shape = input.shape
    input_strides = input.stride()

    # Compute the number of elements in the input tensor
    n_elements = input.numel()

    # Compute the shape of the output tensor
    if keepdim:
        output_shape = list(input_shape)
        output_shape[dim] = 1
    else:
        output_shape = list(input_shape)
        del output_shape[dim]

    # Allocate the output tensor
    output = torch.empty(output_shape, dtype=torch.long, device=input.device)

    # Determine the block size
    BLOCK_SIZE = 1024

    # Compute the number of blocks
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Enqueue the kernel
    argmax_kernel[(n_blocks,)](
        output,
        input,
        input_shape,
        input_strides,
        dim,
        keepdim,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example 1: argmax along a specified dimension
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=1)
print(output_tensor)  # Expected output: tensor([[2], [2]])

# Example 2: argmax of the flattened input tensor
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=None)
print(output_tensor)  # Expected output: tensor(5)

# Example 3: argmax with keepdim=True
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=1, keepdim=True)
print(output_tensor)  # Expected output: tensor([[2], [2]])
