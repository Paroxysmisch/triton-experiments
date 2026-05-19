import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_maxpool_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C, H, W, out_C, out_H, out_W,
    conv_kH, conv_kW, conv_stride, conv_padding, conv_dilation, conv_groups,
    pool_kH, pool_kW, pool_stride, pool_padding, pool_dilation, pool_ceil_mode,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (out_H * out_W)
    h = (pid % (out_H * out_W)) // out_W
    w = (pid % (out_H * out_W)) % out_W

    offset_output = batch_id * out_C * out_H * out_W + h * out_W + w
    output_block_ptr = output_ptr + offset_output * BLOCK_SIZE_C

    for c in range(0, out_C, BLOCK_SIZE_C):
        output_block = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
        for kh in range(conv_kH):
            for kw in range(conv_kW):
                for ic in range(C // conv_groups):
                    input_h = h * pool_stride + kh * conv_dilation - conv_padding
                    input_w = w * pool_stride + kw * conv_dilation - conv_padding
                    input_h = max(0, min(H - 1, input_h))
                    input_w = max(0, min(W - 1, input_w))

                    input_block_ptr = input_ptr + batch_id * C * H * W + ic * H * W + input_h * W + input_w
                    weight_block_ptr = weight_ptr + c * (C // conv_groups) * conv_kH * conv_kW + ic * conv_kH * conv_kW + kh * conv_kW + kw

                    input_block = tl.load(input_block_ptr, mask=input_h < H and input_w < W, other=0.0)
                    weight_block = tl.load(weight_block_ptr)

                    output_block += input_block * weight_block

        if bias_ptr is not None:
            bias_block_ptr = bias_ptr + c
            bias_block = tl.load(bias_block_ptr)
            output_block += bias_block

        output_block = tl.max(output_block, 0.0)  # ReLU

        tl.store(output_block_ptr + c, output_block, mask=c < out_C)

# Python Wrapper Function
def relu_max_pool2d_conv2d(
    input, weight, bias=None, conv_stride=1, conv_padding=0, conv_dilation=1, conv_groups=1,
    pool_kernel_size=2, pool_stride=None, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False
):
    if pool_stride is None:
        pool_stride = pool_kernel_size

    B, C, H, W = input.shape
    out_C, _, conv_kH, conv_kW = weight.shape

    if isinstance(conv_stride, int):
        conv_stride = (conv_stride, conv_stride)
    if isinstance(conv_padding, int):
        conv_padding = (conv_padding, conv_padding)
    if isinstance(conv_dilation, int):
        conv_dilation = (conv_dilation, conv_dilation)

    if isinstance(pool_kernel_size, int):
        pool_kernel_size = (pool_kernel_size, pool_kernel_size)
    if isinstance(pool_stride, int):
        pool_stride = (pool_stride, pool_stride)
    if isinstance(pool_padding, int):
        pool_padding = (pool_padding, pool_padding)
    if isinstance(pool_dilation, int):
        pool_dilation = (pool_dilation, pool_dilation)

    out_H = (H + 2 * conv_padding[0] - conv_dilation[0] * (conv_kH - 1) - 1) // conv_stride[0] + 1
    out_W = (W + 2 * conv_padding[1] - conv_dilation[1] * (conv_kW - 1) - 1) // conv_stride[1] + 1

    if pool_ceil_mode:
        out_H = (H + 2 * pool_padding[0] - pool_dilation[0] * (pool_kernel_size[0] - 1) - 1 + pool_stride[0] - 1) // pool_stride[0] + 1
        out_W = (W + 2 * pool_padding[1] - pool_dilation[1] * (pool_kernel_size[1] - 1) - 1 + pool_stride[1] - 1) // pool_stride[1] + 1
    else:
        out_H = (H + 2 * pool_padding[0] - pool_dilation[0] * (pool_kernel_size[0] - 1) - 1) // pool_stride[0] + 1
        out_W = (W + 2 * pool_padding[1] - pool_dilation[1] * (pool_kernel_size[1] - 1) - 1) // pool_stride[1] + 1

    output = torch.empty((B, out_C, out_H, out_W), device=input.device, dtype=input.dtype)

    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16

    grid = (B * out_H * out_W,)

    conv2d_maxpool_relu_kernel[grid](
        input, weight, bias, output,
        B, C, H, W, out_C, out_H, out_W,
        conv_kH, conv_kW, conv_stride[0], conv_padding[0], conv_dilation[0], conv_groups,
        pool_kernel_size[0], pool_kernel_size[1], pool_stride[0], pool_padding[0], pool_dilation[0], pool_ceil_mode,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    if inplace:
        input.copy_(output)
    else:
        return output

import torch
import torch.nn.functional as F

# Example input
input = torch.randn(1, 3, 10, 10, device='cuda')
weight = torch.randn(6, 3, 3, 3, device='cuda')
bias = torch.randn(6, device='cuda')

# PyTorch reference
conv_out = F.conv2d(input, weight, bias, stride=1, padding=1, dilation=1, groups=1)
pool_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
relu_out = F.relu(pool_out, inplace=False)

# Triton implementation
triton_out = relu_max_pool2d_conv2d(input, weight, bias, conv_stride=1, conv_padding=1, conv_dilation=1, conv_groups=1, pool_kernel_size=2, pool_stride=2, pool_padding=0, pool_dilation=1, pool_ceil_mode=False, inplace=False)

# Verify the results
print("PyTorch output:", relu_out)
print("Triton output:", triton_out)
print("Difference:", torch.abs(relu_out - triton_out).max())
