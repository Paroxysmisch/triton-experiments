import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Total number of elements in the input tensor
    stride_dim,  # Stride along the specified dimension
    stride_batch,  # Stride along the batch dimension
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Compute the block index
    pid = tl.program_id(axis=0)
    # Compute the batch index
    batch_idx = pid * BLOCK_SIZE
    # Iterate over the elements in the block
    for i in range(BLOCK_SIZE):
        if batch_idx + i < n_elements:
            # Load the input element
            input_val = tl.load(input_ptr + (batch_idx + i) * stride_batch)
            # Apply the natural logarithm
            log_val = tl.log(input_val)
            # Store the log value in a temporary buffer
            temp_val = tl.store(temp_ptr + i, log_val)
    
    # Compute the maximum value in the batch
    max_val = tl.max(temp_val, axis=0)
    # Subtract the maximum value for numerical stability
    for i in range(BLOCK_SIZE):
        if batch_idx + i < n_elements:
            log_val = tl.load(temp_ptr + i)
            log_val = log_val - max_val
            # Compute the exponential of the log value
            exp_val = tl.exp(log_val)
            # Store the exponential value in a temporary buffer
            temp_val = tl.store(temp_ptr + i, exp_val)
    
    # Compute the sum of the exponential values
    sum_val = tl.sum(temp_val, axis=0)
    # Normalize the exponential values to get the softmax
    for i in range(BLOCK_SIZE):
        if batch_idx + i < n_elements:
            exp_val = tl.load(temp_ptr + i)
            softmax_val = exp_val / sum_val
            # Store the result in the output tensor
            tl.store(output_ptr + (batch_idx + i) * stride_batch, softmax_val)

import torch
import triton
import triton.language as tl

def softmax_log(input, dim=-1, dtype=None) -> torch.Tensor:
    # Cast the input tensor to the specified data type if provided
    if dtype is not None:
        input = input.to(dtype)
    
    # Get the shape and strides of the input tensor
    shape = input.shape
    strides = input.stride()
    
    # Ensure the specified dimension is within the valid range
    if dim < -len(shape) or dim >= len(shape):
        raise ValueError(f"Dimension {dim} is out of range for input tensor with shape {shape}")
    
    # Normalize the dimension to be positive
    dim = dim if dim >= 0 else len(shape) + dim
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    
    # Allocate the output tensor with the same shape and data type as the input tensor
    output = torch.empty_like(input)
    
    # Compute the strides for the specified dimension and the batch dimension
    stride_dim = strides[dim]
    stride_batch = strides[0] if dim > 0 else 1
    
    # Launch the Triton kernel
    grid = (n_elements // 1024 + 1,)
    softmax_log_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        stride_dim=stride_dim,
        stride_batch=stride_batch,
        BLOCK_SIZE=1024
    )
    
    return output

import torch

# Define input tensor
input = torch.rand(3, 4) * 10

# Apply softmax_log along the last dimension
result = softmax_log(input, dim=1)
print(result)

# Apply softmax_log along a different dimension
result = softmax_log(input, dim=0)
print(result)
