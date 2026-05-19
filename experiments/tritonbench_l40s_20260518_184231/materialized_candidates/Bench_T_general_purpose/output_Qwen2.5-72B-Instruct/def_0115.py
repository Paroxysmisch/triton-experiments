import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(
    input_ptr,  # Pointer to the input tensor
    weight_ptr,  # Pointer to the weight matrix
    bias_ptr,  # Pointer to the bias tensor (optional)
    output_ptr,  # Pointer to the output tensor
    in_features,  # Number of input features
    out_features,  # Number of output features
    batch_size,  # Batch size (number of input vectors)
    stride_input,  # Stride for the input tensor
    stride_weight,  # Stride for the weight matrix
    stride_bias,  # Stride for the bias tensor (optional)
    stride_output,  # Stride for the output tensor
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Compute the output for the current batch
    output = tl.zeros((out_features,), dtype=tl.float32)
    for i in range(0, in_features, BLOCK_SIZE):
        input_block = tl.load(input_ptr + batch_idx * stride_input + i, mask=i + tl.arange(0, BLOCK_SIZE) < in_features, other=0.0)
        weight_block = tl.load(weight_ptr + i * stride_weight, mask=i + tl.arange(0, BLOCK_SIZE) < in_features, other=0.0)
        output += tl.dot(input_block, weight_block)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr, mask=tl.arange(0, out_features) < out_features, other=0.0)
        output += bias

    # Apply Tanh activation
    output = (tl.exp(output) - tl.exp(-output)) / (tl.exp(output) + tl.exp(-output))

    # Store the result
    tl.store(output_ptr + batch_idx * stride_output, output)

import torch
import triton
import triton.language as tl

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input, weight, and bias tensors are on the same device
    device = input.device
    assert weight.device == device, "Weight tensor must be on the same device as the input tensor."
    if bias is not None:
        assert bias.device == device, "Bias tensor must be on the same device as the input tensor."

    # Get the shapes
    batch_size, in_features = input.shape
    out_features = weight.shape[0]

    # Allocate the output tensor
    output = torch.empty((batch_size, out_features), device=device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (batch_size, )
    block = (1, )

    # Launch the kernel
    tanh_linear_kernel[grid, block](
        input,  # Pointer to the input tensor
        weight,  # Pointer to the weight matrix
        bias,  # Pointer to the bias tensor (optional)
        output,  # Pointer to the output tensor
        in_features,  # Number of input features
        out_features,  # Number of output features
        batch_size,  # Batch size (number of input vectors)
        input.stride(0),  # Stride for the input tensor
        weight.stride(0),  # Stride for the weight matrix
        bias.stride(0) if bias is not None else 0,  # Stride for the bias tensor (optional)
        output.stride(0),  # Stride for the output tensor
        BLOCK_SIZE=128  # Block size for the kernel
    )

    return output
