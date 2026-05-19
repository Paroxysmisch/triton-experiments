import triton
import triton.language as tl

@triton.jit
def argmax_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_shape,  # Shape of the input tensor
    input_strides,  # Strides of the input tensor
    output_strides,  # Strides of the output tensor
    dim,  # Dimension to reduce
    keepdim,  # Whether to keep the reduced dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the range of indices for the current block
    block_end = block_start + BLOCK_SIZE
    block_end = tl.minimum(block_end, input_shape[0])

    # Initialize the maximum value and its index
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    max_idx = tl.full((1,), -1, dtype=tl.int64)

    for i in range(block_start, block_end):
        # Compute the linear index in the input tensor
        linear_idx = 0
        for d in range(len(input_shape)):
            if d == dim:
                linear_idx += i * input_strides[d]
            else:
                linear_idx += 0 * input_strides[d]  # Placeholder for other dimensions

        # Load the value from the input tensor
        val = tl.load(input_ptr + linear_idx)

        # Update the maximum value and its index
        max_val = tl.where(val > max_val, val, max_val)
        max_idx = tl.where(val > max_val, i, max_idx)

    # Write the result to the output tensor
    if keepdim:
        output_idx = pid * output_strides[0]
    else:
        output_idx = 0
    tl.store(output_ptr + output_idx, max_idx)

import torch
import triton
import triton.language as tl

def argmax(input, dim=None, keepdim=False):
    if dim is None:
        input = input.flatten()
        dim = 0
        keepdim = False

    input_shape = input.shape
    input_strides = input.stride()
    input_dtype = input.dtype

    if dim < 0:
        dim += len(input_shape)

    output_shape = list(input_shape)
    if not keepdim:
        output_shape.pop(dim)
    else:
        output_shape[dim] = 1

    output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    # Compute the number of blocks needed
    block_size = 1024
    num_blocks = (input_shape[dim] + block_size - 1) // block_size

    # Launch the Triton kernel
    argmax_kernel[(num_blocks,)](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_shape=input_shape,
        input_strides=input_strides,
        output_strides=output.stride(),
        dim=dim,
        keepdim=keepdim,
        BLOCK_SIZE=block_size,
    )

    return output

# Example 1: argmax along a specific dimension
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=1)
print(output_tensor)  # Expected: tensor([[2], [2]])

# Example 2: argmax of the flattened tensor
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=None)
print(output_tensor)  # Expected: tensor(5)

# Example 3: argmax with keepdim=True
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
output_tensor = argmax(input_tensor, dim=1, keepdim=True)
print(output_tensor)  # Expected: tensor([[2], [2]])
