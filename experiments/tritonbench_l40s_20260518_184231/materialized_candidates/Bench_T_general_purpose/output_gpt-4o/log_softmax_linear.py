import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, bias_flag, dim, BLOCK_SIZE: tl.constexpr
):
    # Calculate the row and column index for the output
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load input and weight for the current row
    input = tl.load(input_ptr + row_idx * in_features + col_idx, mask=col_idx < in_features, other=0.0)
    weight = tl.load(weight_ptr + col_idx * out_features, mask=col_idx < out_features, other=0.0)

    # Perform the linear transformation: xW^T
    linear_out = tl.dot(input, weight)

    # Add bias if provided
    if bias_flag:
        bias = tl.load(bias_ptr + col_idx, mask=col_idx < out_features, other=0.0)
        linear_out += bias

    # Compute the maximum value for numerical stability
    max_val = tl.max(linear_out, axis=0)

    # Compute exp(linear_out - max_val) and sum it
    exp_out = tl.exp(linear_out - max_val)
    sum_exp = tl.sum(exp_out, axis=0)

    # Compute log_softmax: log(exp_out / sum_exp)
    log_softmax_out = linear_out - max_val - tl.log(sum_exp)

    # Store the result
    tl.store(output_ptr + row_idx * out_features + col_idx, log_softmax_out, mask=col_idx < out_features)

import torch

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Validate dimensions
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight tensor must be 2-dimensional"
    assert input.size(-1) == weight.size(-1), "Input features must match weight's in_features"
    if bias is not None:
        assert bias.size(0) == weight.size(0), "Bias must match weight's out_features"

    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
        weight = weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)

    # Flatten the input if necessary
    input_flat = input.view(-1, input.size(-1))
    out_features = weight.size(0)
    in_features = weight.size(1)

    # Allocate output tensor
    output = torch.empty((input_flat.size(0), out_features), device=input.device, dtype=input.dtype)

    # Determine block size for Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(max(in_features, out_features))

    # Launch Triton kernel
    grid = (input_flat.size(0),)
    log_softmax_linear_kernel[grid](
        input_flat, weight, bias, output,
        in_features, out_features, bias is not None, dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape output to match input's shape with out_features
    output = output.view(*input.shape[:-1], out_features)

    return output
