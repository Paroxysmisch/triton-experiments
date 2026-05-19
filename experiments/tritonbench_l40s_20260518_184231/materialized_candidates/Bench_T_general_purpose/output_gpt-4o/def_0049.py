import triton
import triton.language as tl

@triton.jit
def leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    negative_slope,
    in_channels, out_channels, kernel_h, kernel_w,
    input_h, input_w, output_h, output_w,
    BLOCK_SIZE: tl.constexpr
):
    # Define the grid
    pid = tl.program_id(0)
    # Calculate the output coordinates
    oh = pid // output_w
    ow = pid % output_w

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over the input channels
    for c in range(0, in_channels // groups):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Calculate the input coordinates
                ih = oh * stride - padding + kh * dilation
                iw = ow * stride - padding + kw * dilation
                # Check bounds
                if 0 <= ih < input_h and 0 <= iw < input_w:
                    # Load input and weight
                    input_val = tl.load(input_ptr + (c * input_h + ih) * input_w + iw)
                    weight_val = tl.load(weight_ptr + (c * kernel_h + kh) * kernel_w + kw)
                    # Accumulate
                    acc += input_val * weight_val

    # Apply bias if present
    if bias_ptr:
        acc += tl.load(bias_ptr + pid % out_channels)

    # Apply Leaky ReLU
    acc = tl.maximum(acc, 0) + negative_slope * tl.minimum(acc, 0)

    # Store the result
    tl.store(output_ptr + pid, acc)

import torch

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Input dimensions
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    output_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    output_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, output_h, output_w), device=input.device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = (output_h * output_w, )
    leaky_relu_conv2d_kernel[grid](
        input, weight, bias, output,
        stride, padding, dilation, groups,
        negative_slope,
        in_channels, out_channels, kernel_h, kernel_w,
        input_h, input_w, output_h, output_w,
        BLOCK_SIZE=32
    )

    return output
