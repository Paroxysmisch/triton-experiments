import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input,
    weight,
    bias,
    output,
    input_row_stride,
    weight_row_stride,
    output_row_stride,
    n_cols,
    eps,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load input and weight
    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    w_ptr = weight + offsets
    w = tl.load(w_ptr, mask=offsets < n_cols)

    # Linear transformation
    linear_out = tl.dot(x, w)

    # Apply bias if provided
    if bias is not None:
        b_ptr = bias + offsets
        b = tl.load(b_ptr, mask=offsets < n_cols)
        linear_out += b

    # ReLU activation
    relu_out = tl.relu(linear_out)

    # Layer normalization
    mean = tl.sum(relu_out, axis=0) / N_COLS
    var = tl.sum((relu_out - mean) * (relu_out - mean), axis=0) / N_COLS
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Apply layer normalization
    norm_out = (relu_out - mean) * inv_std

    # Apply elementwise affine transformation if enabled
    if elementwise_affine:
        gamma_ptr = weight + offsets
        gamma = tl.load(gamma_ptr, mask=offsets < n_cols)
        beta_ptr = bias + offsets
        beta = tl.load(beta_ptr, mask=offsets < n_cols)
        norm_out = norm_out * gamma + beta

    # Store the result
    out_ptr = output + prog_id * output_row_stride
    tl.store(out_ptr + offsets, norm_out, mask=offsets < n_cols)

import torch
import triton
from triton.runtime.jit import get_cuda_stream

def fused_layer_norm_relu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, normalized_shape=None, eps: float = 1e-5, elementwise_affine: bool = True) -> torch.Tensor:
    """
    Applies a fused operation consisting of a linear transformation followed by ReLU activation and layer normalization on the input tensor.

    Args:
        input (Tensor): Input tensor with shape (*, in_features).
        weight (Tensor): Weights for the linear transformation, shape (out_features, in_features).
        bias (Tensor, optional): Bias for the linear transformation, shape (out_features).
        normalized_shape (int or list or torch.Size, optional): Shape of the dimensions to normalize.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        elementwise_affine (bool, optional): If True, layer normalization has learnable parameters. Default is True.

    Returns:
        Tensor: Result after applying the linear transformation, ReLU, and layer normalization.
    """
    # Ensure normalized_shape is a list
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    elif isinstance(normalized_shape, torch.Size):
        normalized_shape = list(normalized_shape)

    # Check input dimensions
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    assert weight.shape[1] == in_features, "Weight shape must match input features"
    if bias is not None:
        assert bias.shape[0] == out_features, "Bias shape must match output features"

    # Prepare output tensor
    output = torch.empty(input.shape[:-1] + (out_features,), dtype=input.dtype, device=input.device)

    # Set up kernel parameters
    n_cols = in_features
    BLOCK_N = triton.next_power_of_2(n_cols)
    grid = (input.shape[0],)

    # Launch the kernel
    fused_layer_norm_relu_linear_kernel[grid](
        input,
        weight,
        bias,
        output,
        input.stride(-2),
        weight.stride(-2),
        output.stride(-2),
        n_cols,
        eps,
        n_cols,
        BLOCK_N,
        num_warps=4,
        num_stages=2,
        stream=get_cuda_stream(input.device.index)
    )

    return output

import torch

# Example input tensor
input = torch.randn(4, 5)

# Linear transformation weights
weight = torch.randn(3, 5)

# Bias for linear layer
bias = torch.randn(3)

# Normalized shape
normalized_shape = 3

# Apply fused operation
output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)

# Print the output shape
print(output.shape)  # Expected output shape: (4, 3)
