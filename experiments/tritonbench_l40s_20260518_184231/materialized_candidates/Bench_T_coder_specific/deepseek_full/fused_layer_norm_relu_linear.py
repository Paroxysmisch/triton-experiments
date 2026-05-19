import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear(input, weight, bias, normalized_shape, eps, elementwise_affine):
    # Linear transformation
    x = tl.dot(input, weight)
    # Apply bias if specified
    if bias is not None:
        x = tl.add(x, bias)
    # Reshape x for layer normalization
    x = tl.reshape(x, [1, -1, normalized_shape])
    # Define the mean and variance for layer normalization
    mean = tl.mean(x, axis=1)
    variance = tl.var(x, axis=1)
    # Normalize x
    x = tl.where(x >= 0, (x - mean[:, None, :]) / tl.sqrt(variance[:, None, :] + eps), (x - mean[:, None, :]) / tl.sqrt(variance[:, None, :] + eps))
    # Apply elementwise_affine if specified
    if elementwise_affine:
        weight = tl.reshape(weight, [1, -1, normalized_shape])
        bias = tl.reshape(bias, [1, -1, normalized_shape])
        x = tl.mul(x, weight) + bias
    # Apply ReLU activation
    x = tl.where(x >= 0, x, 0)
    return x

def wrapper_fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    if normalized_shape is None:
        normalized_shape = input.shape[-1]
    return fused_layer_norm_relu_linear(input, weight, bias, normalized_shape, eps, elementwise_affine)
