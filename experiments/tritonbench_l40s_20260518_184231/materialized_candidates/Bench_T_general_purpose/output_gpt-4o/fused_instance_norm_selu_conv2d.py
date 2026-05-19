import triton
import triton.language as tl

@triton.jit
def fused_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, in_channels, out_channels, kernel_h, kernel_w,
    input_h, input_w, output_h, output_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Calculate indices for this block
    pid = tl.program_id(axis=0)
    batch_id = pid // (output_h * output_w)
    hw_id = pid % (output_h * output_w)
    out_h = hw_id // output_w
    out_w = hw_id % output_w

    # Calculate the starting position of the input slice
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, in_channels, BLOCK_K):
        # Load input and weight tiles
        input_tile = tl.load(input_ptr + (batch_id * in_channels + k) * input_h * input_w)
        weight_tile = tl.load(weight_ptr + k * out_channels * kernel_h * kernel_w)

        # Compute partial output
        acc += tl.dot(input_tile, weight_tile)

    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + batch_id * out_channels)

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * output_h * output_w + hw_id, acc)

import torch
import torch.nn.functional as F

def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    # Ensure input is a 4D tensor
    assert input.ndim == 4, "Input tensor must be 4D"

    # Extract dimensions
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Calculate output dimensions
    output_h = (input_h + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    output_w = (input_w + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, output_h, output_w), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (batch_size * output_h * output_w,)
    fused_conv2d_kernel[grid](
        input, weight, bias, output,
        stride, padding, dilation, in_channels, out_channels, kernel_h, kernel_w,
        input_h, input_w, output_h, output_w,
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=16
    )

    # Apply SELU activation
    output = F.selu(output)

    # Apply instance normalization
    if num_features is None:
        num_features = out_channels
    output = F.instance_norm(output, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    return output
