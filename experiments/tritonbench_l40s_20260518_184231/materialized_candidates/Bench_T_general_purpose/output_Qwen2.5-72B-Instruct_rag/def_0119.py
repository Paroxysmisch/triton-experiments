import torch
import triton
import triton.language as tl

# Triton kernel for 2D convolution
@triton.jit
def conv2d_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr, stride, padding, dilation, groups,
    minibatch, in_channels, iH, iW, out_channels, kH, kW, upscale_factor,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks_h = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    num_blocks_w = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    num_blocks = num_blocks_h * num_blocks_w
    block_idx_h = pid // num_blocks_w
    block_idx_w = pid % num_blocks_w

    block_start_h = block_idx_h * BLOCK_SIZE_H
    block_start_w = block_idx_w * BLOCK_SIZE_W

    for group in range(groups):
        for out_c in range(group * out_channels // groups, (group + 1) * out_channels // groups):
            for in_c in range(group * in_channels // groups, (group + 1) * in_channels // groups):
                for kh in range(kH):
                    for kw in range(kW):
                        for bh in range(BLOCK_SIZE_H):
                            for bw in range(BLOCK_SIZE_W):
                                h = block_start_h + bh * stride - padding + kh * dilation
                                w = block_start_w + bw * stride - padding + kw * dilation
                                if 0 <= h < iH and 0 <= w < iW:
                                    input_idx = (pid * BLOCK_SIZE_H * BLOCK_SIZE_W + bh * BLOCK_SIZE_W + bw) * in_channels + in_c
                                    weight_idx = (out_c * in_channels * kH * kW + in_c * kH * kW + kh * kW + kw)
                                    output_idx = (pid * BLOCK_SIZE_H * BLOCK_SIZE_W + bh * BLOCK_SIZE_W + bw) * out_channels + out_c
                                    output_ptr[output_idx] += input_ptr[input_idx] * weight_ptr[weight_idx]

    if bias_ptr is not None:
        for out_c in range(out_channels):
            for bh in range(BLOCK_SIZE_H):
                for bw in range(BLOCK_SIZE_W):
                    output_idx = (pid * BLOCK_SIZE_H * BLOCK_SIZE_W + bh * BLOCK_SIZE_W + bw) * out_channels + out_c
                    output_ptr[output_idx] += bias_ptr[out_c]

# Triton kernel for pixel shuffle
@triton.jit
def pixel_shuffle_kernel(
    input_ptr, output_ptr, minibatch, in_channels, iH, iW, upscale_factor,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid = tl.program_id(0)
    block_start_h = pid // (iW * upscale_factor)
    block_start_w = (pid % (iW * upscale_factor)) // upscale_factor
    block_start_c = (pid % (iW * upscale_factor)) % upscale_factor

    for bh in range(BLOCK_SIZE_H):
        for bw in range(BLOCK_SIZE_W):
            h = block_start_h + bh
            w = block_start_w + bw
            c = block_start_c
            if 0 <= h < iH and 0 <= w < iW:
                input_idx = (pid * BLOCK_SIZE_H * BLOCK_SIZE_W + bh * BLOCK_SIZE_W + bw) * in_channels + c
                output_idx = (pid * BLOCK_SIZE_H * BLOCK_SIZE_W + bh * BLOCK_SIZE_W + bw) * in_channels * upscale_factor * upscale_factor + c * upscale_factor * upscale_factor + h * upscale_factor + w
                output_ptr[output_idx] = input_ptr[input_idx]

# Wrapper function
def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."

    minibatch, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    assert in_channels == in_channels_per_group * groups, "Input channels must match the weight dimensions."
    assert out_channels % groups == 0, "Out channels must be divisible by groups."

    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    # Output shape after convolution
    conv_output = torch.empty((minibatch, out_channels, oH, oW), device=device)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    num_blocks_h = (oH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (oW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (num_blocks_h * num_blocks_w,)

    conv2d_kernel[grid](
        input, conv_output, weight, bias, stride, padding, dilation, groups,
        minibatch, in_channels, iH, iW, out_channels, kH, kW, upscale_factor,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    # Output shape after pixel shuffle
    shuffle_output = torch.empty((minibatch, out_channels // (upscale_factor ** 2), oH * upscale_factor, oW * upscale_factor), device=device)

    num_blocks_h = (oH * upscale_factor + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (oW * upscale_factor + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (num_blocks_h * num_blocks_w,)

    pixel_shuffle_kernel[grid](
        conv_output, shuffle_output, minibatch, out_channels, oH, oW, upscale_factor,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return shuffle_output

import torch

# Test data
input = torch.randn(2, 3, 16, 16).cuda()
weight = torch.randn(12, 3, 3, 3).cuda()
bias = torch.randn(12).cuda()

# Call the function
output = pixel_shuffle_conv2d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1, upscale_factor=2)

# Print the output shape
print(output.shape)  # Expected: (2, 3, 32, 32)
