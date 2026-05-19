import triton
import triton.language as tl

@triton.jit
def conv2d_batch_norm_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    stride, padding, dilation, groups,
    running_mean, running_var, bn_weight, bn_bias,
    training, momentum, eps, inplace,
    input_shape, weight_shape, output_shape,
    BLOCK_SIZE: tl.constexpr
):
    # Extract dimensions
    batch_size, in_channels, in_height, in_width = input_shape
    out_channels, _, kernel_height, kernel_width = weight_shape
    out_height, out_width = output_shape[2], output_shape[3]

    # Compute the grid and block indices
    pid = tl.program_id(axis=0)
    num_blocks = (out_height * out_width) // BLOCK_SIZE
    block_id = pid % num_blocks
    batch_id = pid // num_blocks

    # Compute the output coordinates
    h = (block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) // out_width
    w = (block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) % out_width

    # Compute the input coordinates
    h_in = h * stride - padding
    w_in = w * stride - padding

    # Load the input and weight tensors
    input_block = tl.load(input_ptr + batch_id * in_channels * in_height * in_width + 
                          tl.arange(0, in_channels) * in_height * in_width + 
                          h_in * in_width + w_in, mask=(h_in >= 0) & (h_in < in_height) & (w_in >= 0) & (w_in < in_width), other=0.0)

    weight_block = tl.load(weight_ptr + tl.arange(0, out_channels) * kernel_height * kernel_width + 
                           tl.arange(0, in_channels) * kernel_height * kernel_width + 
                           tl.arange(0, kernel_height) * kernel_width + 
                           tl.arange(0, kernel_width))

    # Perform the convolution
    conv_result = tl.sum(input_block * weight_block, axis=1)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr)
        conv_result += bias

    # Perform batch normalization
    if training:
        # Compute mean and variance
        mean = tl.mean(conv_result, axis=0)
        var = tl.var(conv_result, axis=0)

        # Update running mean and variance
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var

        # Normalize
        normalized = (conv_result - mean) / tl.sqrt(var + eps)
    else:
        # Normalize using running mean and variance
        normalized = (conv_result - running_mean) / tl.sqrt(running_var + eps)

    # Scale and shift
    if bn_weight_ptr is not None and bn_bias_ptr is not None:
        bn_weight = tl.load(bn_weight_ptr)
        bn_bias = tl.load(bn_bias_ptr)
        normalized = normalized * bn_weight + bn_bias

    # Apply ReLU
    if inplace:
        output = tl.where(normalized > 0, normalized, 0.0)
    else:
        output = tl.relu(normalized)

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * out_height * out_width + 
             tl.arange(0, out_channels) * out_height * out_width + 
             h * out_width + w, output)

import torch
import triton
import triton.language as tl

def relu_batch_norm_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, running_mean=None, running_var=None, bn_weight=None, bn_bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Get input and weight shapes
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    # Compute output shape
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output_shape = (batch_size, out_channels, out_height, out_width)

    # Allocate output tensor
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Convert stride, padding, and dilation to tuples if necessary
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Launch the Triton kernel
    grid = (batch_size * out_height * out_width // 1024, 1, 1)
    conv2d_batch_norm_relu_kernel[grid](
        input, weight, bias, output,
        running_mean, running_var, bn_weight, bn_bias,
        stride[0], padding[0], dilation[0], groups,
        running_mean, running_var, bn_weight, bn_bias,
        training, momentum, eps, inplace,
        input.shape, weight.shape, output.shape,
        BLOCK_SIZE=1024
    )

    return output
