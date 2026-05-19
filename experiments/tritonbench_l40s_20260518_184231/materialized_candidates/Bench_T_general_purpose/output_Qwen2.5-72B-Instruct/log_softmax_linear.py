import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_in_features, output_features,
    stride_input_batch, stride_input_in_features,
    stride_weight_out_features, stride_weight_in_features,
    stride_output_batch, stride_output_features,
    dim, BLOCK_SIZE: tl.constexpr
):
    # Compute the row index for the current thread
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Check if the current thread is within bounds
    if row_idx >= input_batch or col_idx >= output_features:
        return

    # Initialize the linear output
    linear_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load the bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_idx)

    # Load the input and weight data
    input_row = tl.load(input_ptr + row_idx * stride_input_batch + tl.arange(0, BLOCK_SIZE) * stride_input_in_features, mask=tl.arange(0, BLOCK_SIZE) < input_in_features, other=0.0)
    weight_col = tl.load(weight_ptr + col_idx * stride_weight_out_features + tl.arange(0, BLOCK_SIZE) * stride_weight_in_features, mask=tl.arange(0, BLOCK_SIZE) < input_in_features, other=0.0)

    # Perform the dot product
    linear_output = tl.dot(input_row, weight_col)

    # Add the bias if provided
    if bias_ptr is not None:
        linear_output += bias

    # Compute the max value for numerical stability
    max_val = tl.max(linear_output, axis=0)

    # Subtract the max value for numerical stability
    linear_output -= max_val

    # Compute the log-sum-exp
    log_sum_exp = tl.log(tl.sum(tl.exp(linear_output), axis=0))

    # Compute the log-softmax
    log_softmax_output = linear_output - log_sum_exp

    # Store the result
    tl.store(output_ptr + row_idx * stride_output_batch + col_idx * stride_output_features, log_softmax_output)

import torch
import triton
import triton.language as tl

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Cast input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)

    # Ensure the input and weight dimensions are compatible
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions."
    assert input.size(-1) == weight.size(1), "Incompatible input and weight dimensions."

    # Get the dimensions
    input_shape = input.shape
    weight_shape = weight.shape
    input_batch = input_shape[:-1]
    input_in_features = input_shape[-1]
    output_features = weight_shape[0]

    # Flatten the input if it has more than 2 dimensions
    if input.dim() > 2:
        input = input.view(-1, input_in_features)

    # Allocate the output tensor
    output = torch.empty((input.size(0), output_features), device=input.device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (input.size(0), output_features)
    block = (1,)

    # Launch the kernel
    log_softmax_linear_kernel[grid, block](
        input, weight, bias, output,
        input.size(0), input_in_features, output_features,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        dim
    )

    # Reshape the output if the input was flattened
    if input.dim() > 2:
        output = output.view(*input_batch, output_features)

    return output
