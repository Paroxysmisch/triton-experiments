import triton
import triton.language as tl

@triton.jit
def sum_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_shape,  # Shape of the input tensor
    input_strides,  # Strides of the input tensor
    output_strides,  # Strides of the output tensor
    dim,  # Dimensions to reduce
    keepdim,  # Whether to keep the reduced dimensions
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
):
    # Get the current block index
    pid = tl.program_id(axis=0)
    
    # Calculate the number of elements in the input tensor
    num_elements = 1
    for size in input_shape:
        num_elements *= size
    
    # Calculate the number of elements in the output tensor
    output_size = 1
    for i, size in enumerate(input_shape):
        if i not in dim:
            output_size *= size
        elif keepdim:
            output_size *= 1
    
    # Calculate the starting index for the current block
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, num_elements)
    
    # Initialize the sum to zero
    sum_val = tl.zeros((1,), dtype=tl.float32)
    
    # Iterate over the elements in the current block
    for i in range(block_start, block_end):
        # Calculate the index in the input tensor
        input_index = 0
        for j, stride in enumerate(input_strides):
            if j not in dim:
                input_index += (i // (output_size // input_shape[j])) * stride
            else:
                input_index += (i % input_shape[j]) * stride
        
        # Load the value from the input tensor
        value = tl.load(input_ptr + input_index)
        
        # Accumulate the sum
        sum_val += value
    
    # Write the sum to the output tensor
    output_index = pid
    tl.store(output_ptr + output_index, sum_val)

import torch
import triton
import triton.language as tl

def sum(input, dim=None, keepdim=False, *, dtype=None):
    # Convert the input tensor to the desired data type if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Get the shape and strides of the input tensor
    input_shape = input.shape
    input_strides = input.stride()
    
    # Determine the dimensions to reduce
    if dim is None:
        dim = list(range(len(input_shape)))
    elif isinstance(dim, int):
        dim = [dim]
    
    # Calculate the shape of the output tensor
    output_shape = list(input_shape)
    for d in dim:
        if keepdim:
            output_shape[d] = 1
        else:
            output_shape[d] = 0
    output_shape = [s for s in output_shape if s != 0]
    
    # Create the output tensor
    output = torch.zeros(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch the Triton kernel
    grid = (output.numel(),)
    sum_kernel[grid](
        input.data_ptr(),  # Pointer to the input tensor
        output.data_ptr(),  # Pointer to the output tensor
        input_shape,  # Shape of the input tensor
        input_strides,  # Strides of the input tensor
        output.stride(),  # Strides of the output tensor
        dim,  # Dimensions to reduce
        keepdim,  # Whether to keep the reduced dimensions
        BLOCK_SIZE=1024,  # Block size for parallelism
    )
    
    return output

# Example input tensor
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Test the sum function
result = sum(input_tensor, dim=1, keepdim=True)
print(result)  # Expected output: [[6.], [15.]]

result = sum(input_tensor, dim=0, keepdim=False)
print(result)  # Expected output: [5., 7., 9.]

result = sum(input_tensor, dim=[0, 1], keepdim=False)
print(result)  # Expected output: 21.
