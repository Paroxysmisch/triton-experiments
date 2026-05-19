import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    in_channels, out_channels, groups, iH, iW, kH, kW, oH, oW,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_IN: tl.constexpr, BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = oH * oW // BLOCK_SIZE_H
    num_pid_n = out_channels // BLOCK_SIZE_OUT
    num_pid_k = in_channels // BLOCK_SIZE_IN

    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = (pid // num_pid_m) // num_pid_n

    offs_b = tl.arange(0, BLOCK_SIZE_BATCH)
    offs_m = pid_m * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_n = pid_n * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offs_k = pid_k * BLOCK_SIZE_IN + tl.arange(0, BLOCK_SIZE_IN)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(BLOCK_SIZE_BATCH, in_channels, iH, iW),
        strides=(in_channels * iH * iW, iH * iW, iW, 1),
        offsets=(offs_b, offs_k, 0, 0),
        block_shape=(BLOCK_SIZE_BATCH, BLOCK_SIZE_IN, BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(0, 1, 2, 3)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(out_channels, in_channels // groups, kH, kW),
        strides=(in_channels // groups * kH * kW, kH * kW, kW, 1),
        offsets=(offs_n, offs_k, 0, 0),
        block_shape=(BLOCK_SIZE_OUT, BLOCK_SIZE_IN, kH, kW),
        order=(0, 1, 2, 3)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(BLOCK_SIZE_BATCH, out_channels, oH, oW),
        strides=(out_channels * oH * oW, oH * oW, oW, 1),
        offsets=(offs_b, offs_n, offs_m, 0),
        block_shape=(BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(0, 1, 2, 3)
    )

    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for k in range(0, num_pid_k * BLOCK_SIZE_IN, BLOCK_SIZE_IN):
        input = tl.load(input_block_ptr)
        weight = tl.load(weight_block_ptr)

        acc += tl.dot(input, weight)

        input_block_ptr = tl.advance(input_block_ptr, [0, BLOCK_SIZE_IN, 0, 0])
        weight_block_ptr = tl.advance(weight_block_ptr, [0, BLOCK_SIZE_IN, 0, 0])

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n)
        acc += bias

    tl.store(output_block_ptr, acc)

import torch
import triton
import triton.language as tl

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Ensure input, weight, and bias are on the same device
    device = input.device
    input = input.to(device)
    weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Convert stride, padding, and dilation to tuples if they are single numbers
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Extract dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    in_channels_per_group *= groups

    # Compute output dimensions
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Allocate output tensor
    output = torch.empty((minibatch, out_channels, oH, oW), device=device, dtype=input.dtype)

    # Launch the Triton kernel
    grid = lambda META: (
        (oH * oW) // META['BLOCK_SIZE_H'],
        out_channels // META['BLOCK_SIZE_OUT'],
        in_channels // META['BLOCK_SIZE_IN']
    )

    conv2d_kernel[grid](
        input, weight, bias, output,
        stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
        in_channels, out_channels, groups, iH, iW, kH, kW, oH, oW,
        BLOCK_SIZE_BATCH=1, BLOCK_SIZE_IN=32, BLOCK_SIZE_OUT=32,
        BLOCK_SIZE_H=16, BLOCK_SIZE_W=16
    )

    return output
