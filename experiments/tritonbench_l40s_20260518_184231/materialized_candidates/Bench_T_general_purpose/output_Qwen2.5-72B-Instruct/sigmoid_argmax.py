import triton
import triton.language as tl

@triton.jit
def sigmoid_argmax_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_size,  # Total number of elements in the input tensor
    dim,  # Dimension to reduce (None if reducing over all elements)
    keepdim,  # Whether to keep the reduced dimension
    stride,  # Stride of the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the sigmoid for each element in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    sigmoid_vals = 1 / (1 + tl.exp(-input_vals))

    # Find the index of the maximum value in the block
    max_val, max_idx = tl.max_argmax(sigmoid_vals, axis=0, mask=mask)

    # Write the maximum index to the output tensor
    if keepdim:
        output_idx = pid
    else:
        output_idx = 0
    tl.store(output_ptr + output_idx, max_idx, mask=mask)

import torch
import triton
import triton.language as tl

def sigmoid_argmax(input, dim=None, keepdim=False):
    # Flatten the input tensor if dim is None
    if dim is None:
        input = input.flatten()
        dim = 0
        keepdim = False

    # Determine the size of the output tensor
    output_shape = list(input.shape)
    if not keepdim:
        output_shape.pop(dim)
    else:
        output_shape[dim] = 1
    output = torch.empty(output_shape, dtype=torch.long, device=input.device)

    # Launch the Triton kernel
    grid = (input.numel() // 1024 + 1,)
    sigmoid_argmax_kernel[grid](
        input,  # Pointer to the input tensor
        output,  # Pointer to the output tensor
        input.numel(),  # Total number of elements in the input tensor
        dim,  # Dimension to reduce
        keepdim,  # Whether to keep the reduced dimension
        input.stride(dim),  # Stride of the input tensor
        BLOCK_SIZE=1024  # Block size for parallelization
    )

    return output

# Example usage
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')

# Test with dim=None
result1 = sigmoid_argmax(input_tensor, dim=None)
print("Result with dim=None:", result1)  # Expected: 5 (index of the maximum value in the flattened tensor)

# Test with dim=0
result2 = sigmoid_argmax(input_tensor, dim=0)
print("Result with dim=0:", result2)  # Expected: [1, 1, 1] (indices of the maximum values along dim=0)

# Test with dim=1
result3 = sigmoid_argmax(input_tensor, dim=1)
print("Result with dim=1:", result3)  # Expected: [2, 2] (indices of the maximum values along dim=1)

# Test with dim=1 and keepdim=True
result4 = sigmoid_argmax(input_tensor, dim=1, keepdim=True)
print("Result with dim=1 and keepdim=True:", result4)  # Expected: [[2], [2]]
