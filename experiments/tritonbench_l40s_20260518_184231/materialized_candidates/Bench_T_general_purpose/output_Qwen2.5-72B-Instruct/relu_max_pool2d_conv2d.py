import triton
import triton.language as tl

@triton.jit
def conv2d_max_pool2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride, weight_stride, output_stride,
    input_shape, weight_shape, output_shape,
    conv_stride, conv_padding, conv_dilation, conv_groups,
    pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    pid = tl.program_id(axis=0)
    batch_size, in_channels, iH, iW = input_shape
    out_channels, _, kH, kW = weight_shape
    _, _, oH, oW = output_shape

    # Compute the output coordinates for this block
    n = pid // (oH * oW)
    h = (pid % (oH * oW)) // oW
    w = (pid % (oH * oW)) % oW

    # Initialize the output value
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute the convolution
    for kh in range(kH):
        for kw in range(kW):
            for c in range(in_channels // conv_groups):
                # Compute the input coordinates
                ih = h * conv_stride - conv_padding + kh * conv_dilation
                iw = w * conv_stride - conv_padding + kw * conv_dilation

                # Check if the input coordinates are within bounds
                if 0 <= ih < iH and 0 <= iw < iW:
                    # Load the input and weight values
                    input_val = tl.load(input_ptr + n * input_stride[0] + c * input_stride[1] + ih * input_stride[2] + iw * input_stride[3])
                    weight_val = tl.load(weight_ptr + (c * kH * kW + kh * kW + kw) * weight_stride[1] + (pid % out_channels) * weight_stride[0])

                    # Perform the convolution
                    out += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + (pid % out_channels))
        out += bias_val

    # Perform max pooling
    pool_h_start = h * pool_stride - pool_padding
    pool_w_start = w * pool_stride - pool_padding
    pool_h_end = pool_h_start + pool_kernel_size
    pool_w_end = pool_w_start + pool_kernel_size

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    for ph in range(pool_h_start, pool_h_end, pool_dilation):
        for pw in range(pool_w_start, pool_w_end, pool_dilation):
            if 0 <= ph < oH and 0 <= pw < oW:
                max_val = tl.maximum(max_val, out)

    # Apply ReLU
    max_val = tl.maximum(max_val, 0)

    # Store the result
    tl.store(output_ptr + n * output_stride[0] + (pid % out_channels) * output_stride[1] + h * output_stride[2] + w * output_stride[3], max_val)

import torch
import triton
import triton.language as tl

def relu_max_pool2d_conv2d(input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False):
    # Ensure input and weight are on the same device
    device = input.device
    input = input.to(device)
    weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Compute the output shape
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    if pool_stride is None:
        pool_stride = pool_kernel_size

    oH = (iH + 2 * conv_padding - (kH - 1) * conv_dilation - 1) // conv_stride + 1
    oW = (iW + 2 * conv_padding - (kW - 1) * conv_dilation - 1) // conv_stride + 1

    if pool_ceil_mode:
        oH = (iH + 2 * conv_padding - (kH - 1) * conv_dilation - 1 + conv_stride - 1) // conv_stride + 1
        oW = (iW + 2 * conv_padding - (kW - 1) * conv_dilation - 1 + conv_stride - 1) // conv_stride + 1

    oH = (oH + 2 * pool_padding - (pool_kernel_size - 1) * pool_dilation - 1) // pool_stride + 1
    oW = (oW + 2 * pool_padding - (pool_kernel_size - 1) * pool_dilation - 1) // pool_stride + 1

    output_shape = (batch_size, out_channels, oH, oW)

    # Allocate the output tensor
    output = torch.empty(output_shape, device=device, dtype=input.dtype)

    # Compute the strides
    input_stride = input.stride()
    weight_stride = weight.stride()
    output_stride = output.stride()

    # Launch the Triton kernel
    grid = (batch_size * out_channels * oH * oW, )
    conv2d_max_pool2d_relu_kernel[grid](
        input, weight, bias, output,
        input_stride, weight_stride, output_stride,
        input.shape, weight.shape, output.shape,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        pool_kernel_size, pool_stride, pool_padding, pool_dilation, pool_ceil_mode,
        BLOCK_SIZE=128
    )

    return output

import torch

# Sample input data
input = torch.randn(2, 3, 10, 10, device='cuda')
weight = torch.randn(4, 3, 3, 3, device='cuda')
bias = torch.randn(4, device='cuda')

# Reference implementation using PyTorch
conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
conv.weight.data = weight
conv.bias.data = bias
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
output_ref = torch.nn.functional.relu(pool(conv(input)))

# Triton implementation
output_triton = relu_max_pool2d_conv2d(input, weight, bias, conv_stride=1, conv_padding=1, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=2, pool_padding=0, pool_dilation=1, pool_ceil_mode=False)

# Verify the results
print("Reference output:\n", output_ref)
print("Triton output:\n", output_triton)
print("Difference:\n", torch.abs(output_ref - output_triton).max())
