import triton
import triton.language as tl
import torch

@triton.jit
def leaky_relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width, weight_height, weight_width,
    stride, padding, dilation, negative_slope,
    output_height, output_width, num_channels, BLOCK_SIZE: tl.constexpr
):
    # Define offsets
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize output
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over input channels
    for c in range(num_channels):
        for i in range(weight_height):
            for j in range(weight_width):
                # Compute input indices
                row = row_idx * stride + i * dilation - padding
                col = col_idx * stride + j * dilation - padding

                # Load input and weights
                input_val = tl.load(input_ptr + c * input_height * input_width + row * input_width + col, mask=(row >= 0) & (row < input_height) & (col >= 0) & (col < input_width), other=0.0)
                weight_val = tl.load(weight_ptr + c * weight_height * weight_width + i * weight_width + j)

                # Perform convolution
                output += input_val * weight_val

    # Apply bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE))
        output += bias_val

    # Apply Leaky ReLU
    output = tl.where(output >= 0, output, negative_slope * output)

    # Store result
    tl.store(output_ptr + row_idx * output_width + col_idx, output)

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Ensure input and weight are compatible
    assert input.shape[1] == weight.shape[1] * groups, "Input and weight channels mismatch"

    # Compute output dimensions
    output_height = (input.shape[2] + 2 * padding - dilation * (weight.shape[2] - 1) - 1) // stride + 1
    output_width = (input.shape[3] + 2 * padding - dilation * (weight.shape[3] - 1) - 1) // stride + 1

    # Prepare output tensor
    output = torch.empty((input.shape[0], weight.shape[0], output_height, output_width), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (triton.cdiv(output_height, 32), triton.cdiv(output_width, 32))
    leaky_relu_conv2d_kernel[grid](
        input, weight, bias, output,
        input.shape[2], input.shape[3], weight.shape[2], weight.shape[3],
        stride, padding, dilation, negative_slope,
        output_height, output_width, input.shape[1],
        BLOCK_SIZE=32
    )

    return output
