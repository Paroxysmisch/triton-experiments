import triton
import triton.language as tl

@triton.jit
def leaky_relu(x, negative_slope):
    """
    LeakyReLU activation function
    """
    zero = 0.0
    return tl.where(x >= 0, x, negative_slope * x)

@triton.jit
def conv2d_leaky_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    input_shape, weight_shape, output_shape,
    negative_slope,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) // (BLOCK_SIZE_N * BLOCK_SIZE_C * BLOCK_SIZE_H * BLOCK_SIZE_W)
    pid_n = pid // (num_blocks // output_shape[0])
    pid = pid % (num_blocks // output_shape[0])
    pid_c = pid // (num_blocks // (output_shape[0] * output_shape[1]))
    pid = pid % (num_blocks // (output_shape[0] * output_shape[1]))
    pid_h = pid // (num_blocks // (output_shape[0] * output_shape[1] * output_shape[2]))
    pid_w = pid % (num_blocks // (output_shape[0] * output_shape[1] * output_shape[2]))

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=input_shape,
        strides=(input_shape[1] * input_shape[2] * input_shape[3], input_shape[2] * input_shape[3], input_shape[3], 1),
        offsets=(pid_n, pid_c, pid_h * stride[0] - padding[0], pid_w * stride[1] - padding[1]),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(0, 1, 2, 3)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=weight_shape,
        strides=(weight_shape[1] * weight_shape[2] * weight_shape[3], weight_shape[2] * weight_shape[3], weight_shape[3], 1),
        offsets=(0, 0, 0, 0),
        block_shape=(BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W, 1),
        order=(0, 1, 2, 3)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=output_shape,
        strides=(output_shape[1] * output_shape[2] * output_shape[3], output_shape[2] * output_shape[3], output_shape[3], 1),
        offsets=(pid_n, pid_c, pid_h, pid_w),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(0, 1, 2, 3)
    )

    output = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for i in range(weight_shape[2]):
        for j in range(weight_shape[3]):
            input_block = tl.load(input_block_ptr)
            weight_block = tl.load(weight_block_ptr)
            output += input_block * weight_block
            input_block_ptr = tl.advance(input_block_ptr, (0, 0, dilation[0], dilation[1]))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, 0, 1, 1))

    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr)
        output += bias_block

    output = leaky_relu(output, negative_slope)
    tl.store(output_block_ptr, output)

import torch
import triton
import triton.language as tl

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)

    N, C, H, W = input.shape
    K, C_in, R, S = weight.shape
    assert C_in * groups == C, "Input channels must match weight channels * groups"
    assert K % groups == 0, "Output channels must be divisible by groups"

    H_out = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1

    output_shape = (N, K, H_out, W_out)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(N * K * H_out * W_out, BLOCK_SIZE_N * BLOCK_SIZE_C * BLOCK_SIZE_H * BLOCK_SIZE_W),)
    conv2d_leaky_relu_kernel[grid](
        input, weight, bias, output,
        stride, padding, dilation, groups,
        input.shape, weight.shape, output.shape,
        negative_slope,
        BLOCK_SIZE_N=16, BLOCK_SIZE_C=16, BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
    )

    if inplace:
        input.copy_(output)
        return input
    else:
        return output
