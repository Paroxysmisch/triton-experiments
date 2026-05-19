import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, normalized_shape, eps,
    stride_in, stride_out, stride_weight, stride_bias,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE

    # Load the input and weight
    input = tl.load(input_ptr + batch_start * stride_in, mask=batch_start + tl.arange(0, BLOCK_SIZE) < in_features, other=0.0)
    weight = tl.load(weight_ptr + batch_start * stride_weight, mask=batch_start + tl.arange(0, BLOCK_SIZE) < out_features * in_features, other=0.0)

    # Linear transformation
    linear_output = tl.dot(input, weight)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr, mask=batch_start + tl.arange(0, BLOCK_SIZE) < out_features, other=0.0)
        linear_output += bias

    # ReLU activation
    relu_output = tl.where(linear_output > 0, linear_output, 0)

    # Layer normalization
    mean = tl.sum(relu_output, axis=1) / normalized_shape
    var = tl.sum((relu_output - mean[:, None]) ** 2, axis=1) / normalized_shape
    inv_std = 1 / tl.sqrt(var + eps)
    normalized_output = (relu_output - mean[:, None]) * inv_std[:, None]

    # Store the result
    tl.store(output_ptr + batch_start * stride_out, normalized_output, mask=batch_start + tl.arange(0, BLOCK_SIZE) < out_features)

import torch
import triton
import triton.language as tl

def fused_layer_norm_relu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, normalized_shape=None, eps=1e-5, elementwise_affine=True) -> torch.Tensor:
    # Ensure input and weight are on the same device
    device = input.device
    input = input.to(device)
    weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Ensure normalized_shape is valid
    if normalized_shape is None:
        normalized_shape = weight.shape[0]

    # Allocate output tensor
    output = torch.empty(input.shape[:-1] + (weight.shape[0],), device=device)

    # Launch the Triton kernel
    grid = (input.shape[0],)
    fused_layer_norm_relu_linear_kernel[grid](
        input, weight, bias, output,
        input.shape[-1], weight.shape[0], normalized_shape, eps,
        input.stride(-1), output.stride(-1), weight.stride(-1), bias.stride(-1) if bias is not None else 0,
        BLOCK_SIZE=1024
    )

    return output

# Example usage
if __name__ == "__main__":
    input = torch.randn(4, 5)  # Example input tensor
    weight = torch.randn(3, 5)  # Linear transformation weights
    bias = torch.randn(3)  # Bias for linear layer
    normalized_shape = 3  # Normalized shape for layer normalization

    # Apply fused operation
    output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
    print(output.shape)  # Expected output shape: (4, 3)

if __name__ == "__main__":
    input = torch.randn(4, 5)  # Example input tensor
    weight = torch.randn(3, 5)  # Linear transformation weights
    bias = torch.randn(3)  # Bias for linear layer
    normalized_shape = 3  # Normalized shape for layer normalization

    # Apply fused operation
    output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
    print(output.shape)  # Expected output shape: (4, 3)
