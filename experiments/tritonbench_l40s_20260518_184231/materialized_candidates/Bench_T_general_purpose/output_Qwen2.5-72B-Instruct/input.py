import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_shape,  # Shape of the input tensor
    input_strides,  # Strides of the input tensor
    output_strides,  # Strides of the output tensor
    dim,  # Dimensions to reduce
    keepdim,  # Whether to keep the reduced dimensions
    dtype,  # Data type of the input tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    num_elements = input_shape[0] * input_shape[1]
    block_start = pid * BLOCK_SIZE

    # Initialize the sum and count
    sum_val = tl.zeros((BLOCK_SIZE,), dtype=dtype)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < num_elements:
            input_idx = [0] * len(input_shape)
            for d in range(len(input_shape)):
                if d in dim:
                    input_idx[d] = 0
                else:
                    input_idx[d] = idx // input_strides[d]
                    idx %= input_strides[d]
            input_val = tl.load(input_ptr + tl.sum(input_idx[d] * input_strides[d] for d in range(len(input_shape))))
            sum_val[i] += input_val
            count[i] += 1

    # Reduce the sum and count within the block
    sum_val = tl.sum(sum_val)
    count = tl.sum(count)

    # Write the result to the output tensor
    output_idx = [0] * len(output_strides)
    for d in range(len(output_strides)):
        if d in dim and keepdim:
            output_idx[d] = 0
        else:
            output_idx[d] = pid // output_strides[d]
            pid %= output_strides[d]
    tl.store(output_ptr + tl.sum(output_idx[d] * output_strides[d] for d in range(len(output_strides))), sum_val / count)

import torch
import triton
import triton.language as tl

def mean(input, dim, keepdim=False, dtype=None, out=None):
    # Ensure the input tensor is of the specified data type
    if dtype is not None:
        input = input.to(dtype)

    # Determine the shape and strides of the input tensor
    input_shape = input.shape
    input_strides = input.stride()

    # Determine the shape and strides of the output tensor
    if isinstance(dim, int):
        dim = [dim]
    output_shape = list(input_shape)
    for d in dim:
        if keepdim:
            output_shape[d] = 1
        else:
            output_shape.pop(d)
    output_strides = [1] * len(output_shape)
    for i in range(len(output_shape) - 1, 0, -1):
        output_strides[i - 1] = output_strides[i] * output_shape[i]

    # Allocate the output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output tensor shape does not match the expected shape"

    # Launch the Triton kernel
    grid = (out.numel() // 1024 + 1,)
    mean_kernel[grid](
        input.data_ptr(),  # Pointer to the input tensor
        out.data_ptr(),  # Pointer to the output tensor
        input_shape,  # Shape of the input tensor
        input_strides,  # Strides of the input tensor
        output_strides,  # Strides of the output tensor
        dim,  # Dimensions to reduce
        keepdim,  # Whether to keep the reduced dimensions
        input.dtype,  # Data type of the input tensor
        BLOCK_SIZE=1024,  # Block size for parallelization
    )

    return out

# Sample input tensor
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Test the mean function
result = mean(input_tensor, dim=1, keepdim=True)
print(result)  # Expected output: [[2.0], [5.0]]

result = mean(input_tensor, dim=1, keepdim=False)
print(result)  # Expected output: [2.0, 5.0]

result = mean(input_tensor, dim=(0, 1), keepdim=True)
print(result)  # Expected output: [[3.5]]

result = mean(input_tensor, dim=(0, 1), keepdim=False)
print(result)  # Expected output: [3.5]
