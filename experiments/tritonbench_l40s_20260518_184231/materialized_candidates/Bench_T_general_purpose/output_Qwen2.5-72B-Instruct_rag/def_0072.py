import triton
import triton.language as tl

@triton.jit
def relu_batch_norm_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr, stride, padding, dilation, groups, momentum, eps, training, inplace,
    input_shape, weight_shape, output_shape, 
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # Load input and weight tensors
    input = tl.load(input_ptr)
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr) if bias_ptr is not None else None
    running_mean = tl.load(running_mean_ptr) if running_mean_ptr is not None else None
    running_var = tl.load(running_var_ptr) if running_var_ptr is not None else None
    bn_weight = tl.load(bn_weight_ptr) if bn_weight_ptr is not None else None
    bn_bias = tl.load(bn_bias_ptr) if bn_bias_ptr is not None else None

    # Perform 2D convolution
    conv_output = tl.zeros(output_shape, dtype=tl.float32)
    for h in range(BLOCK_SIZE_H):
        for w in range(BLOCK_SIZE_W):
            for c in range(BLOCK_SIZE_C):
                conv_output[:, :, h, w] += tl.dot(input[:, :, h * stride[0] + h, w * stride[1] + w], weight[:, c, :, :])

    # Add bias if provided
    if bias is not None:
        conv_output += bias

    # Perform batch normalization
    if training:
        # Calculate mean and variance
        batch_mean = tl.mean(conv_output, axis=(0, 2, 3))
        batch_var = tl.var(conv_output, axis=(0, 2, 3))

        # Update running mean and variance
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

    # Normalize the output
    norm_output = (conv_output - running_mean) / tl.sqrt(running_var + eps)
    if bn_weight is not None and bn_bias is not None:
        norm_output = norm_output * bn_weight + bn_bias

    # Apply ReLU activation
    if inplace:
        output = tl.where(norm_output > 0, norm_output, 0)
    else:
        output = tl.max(norm_output, 0)

    # Store the output
    tl.store(output_ptr, output)

import torch
import triton
import triton.language as tl

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Ensure input and weight are on the same device
    device = input.device
    input = input.to(device)
    weight = weight.to(device)
    bias = bias.to(device) if bias is not None else None
    running_mean = running_mean.to(device) if running_mean is not None else None
    running_var = running_var.to(device) if running_var is not None else None
    bn_weight = bn_weight.to(device) if bn_weight is not None else None
    bn_bias = bn_bias.to(device) if bn_bias is not None else None

    # Get input and weight shapes
    minibatch, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape

    # Calculate output shape
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    output_shape = (minibatch, out_channels, oH, oW)

    # Allocate output tensor
    output = torch.empty(output_shape, device=device, dtype=input.dtype)

    # Define grid and block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32

    # Call the Triton kernel
    relu_batch_norm_conv2d_kernel[
        (minibatch, out_channels, oH, oW)
    ](
        input, weight, bias, running_mean, running_var, bn_weight, bn_bias,
        output, stride, padding, dilation, groups, momentum, eps, training, inplace,
        input.shape, weight.shape, output.shape,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return output

import torch

# Define input and weight tensors
input = torch.randn(2, 3, 32, 32, device='cuda')
weight = torch.randn(4, 3, 3, 3, device='cuda')
bias = torch.randn(4, device='cuda')
running_mean = torch.randn(4, device='cuda')
running_var = torch.randn(4, device='cuda')
bn_weight = torch.randn(4, device='cuda')
bn_bias = torch.randn(4, device='cuda')

# Call the function
output = relu_batch_norm_conv2d(
    input, weight, bias, stride=1, padding=1, dilation=1, groups=1,
    running_mean=running_mean, running_var=running_var, bn_weight=bn_weight, bn_bias=bn_bias,
    training=True, momentum=0.1, eps=1e-5, inplace=False
)

print(output.shape)  # Should be (2, 4, 32, 32)
