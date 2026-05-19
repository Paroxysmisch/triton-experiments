import triton
import triton.language as tl

@triton.jit
def conv2d_leaky_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride, weight_stride, output_stride,
    N, C, H, W, K, R, S, P, Q,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, negative_slope,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid = tl.program_id(axis=0)
    n = pid // (Q * P)
    q = (pid % (Q * P)) // Q
    p = (pid % (Q * P)) % Q

    output_offset = n * output_stride[0] + q * output_stride[1] + p * output_stride[2]
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + output_offset,
        shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
        strides=output_stride,
        offsets=(0, 0, 0, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
        order=(0, 1, 2, 3)
    )

    for k in range(K):
        for r in range(R):
            for s in range(S):
                input_offset = n * input_stride[0] + (q * stride_h + r * dilation_h - padding_h) * input_stride[1] + (p * stride_w + s * dilation_w - padding_w) * input_stride[2]
                weight_offset = k * weight_stride[0] + r * weight_stride[1] + s * weight_stride[2]

                input_block_ptr = tl.make_block_ptr(
                    base=input_ptr + input_offset,
                    shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
                    strides=input_stride,
                    offsets=(0, 0, 0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
                    order=(0, 1, 2, 3)
                )

                weight_block_ptr = tl.make_block_ptr(
                    base=weight_ptr + weight_offset,
                    shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
                    strides=weight_stride,
                    offsets=(0, 0, 0, 0),
                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
                    order=(0, 1, 2, 3)
                )

                input_block = tl.load(input_block_ptr)
                weight_block = tl.load(weight_block_ptr)
                output_block = tl.load(output_block_ptr)

                output_block += tl.dot(input_block, weight_block)

                tl.store(output_block_ptr, output_block)

    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(
            base=bias_ptr,
            shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
            strides=(1, 1, 1, 1),
            offsets=(0, 0, 0, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W),
            order=(0, 1, 2, 3)
        )
        bias_block = tl.load(bias_block_ptr)
        output_block += bias_block

    output_block = tl.where(output_block > 0, output_block, output_block * negative_slope)
    tl.store(output_block_ptr, output_block)

import torch
import triton
import triton.language as tl

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False) -> torch.Tensor:
    N, C, H, W = input.shape
    K, C_in, R, S = weight.shape
    assert C_in * groups == C, "Input channels must match the weight channels * groups"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    P = (H + 2 * padding_h - dilation_h * (R - 1) - 1) // stride_h + 1
    Q = (W + 2 * padding_w - dilation_w * (S - 1) - 1) // stride_w + 1

    output = torch.empty((N, K, P, Q), device=input.device, dtype=input.dtype)

    input_stride = input.stride()
    weight_stride = weight.stride()
    output_stride = output.stride()

    grid = (N * P * Q,)

    conv2d_leaky_relu_kernel[grid](
        input, weight, bias, output,
        input_stride, weight_stride, output_stride,
        N, C, H, W, K, R, S, P, Q,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
        groups, negative_slope,
        BLOCK_SIZE_N=1, BLOCK_SIZE_C=1, BLOCK_SIZE_H=1, BLOCK_SIZE_W=1
    )

    if inplace:
        input.copy_(output)
        return input
    else:
        return output

# Sample data
input = torch.randn(2, 3, 10, 10).cuda()
weight = torch.randn(4, 3, 3, 3).cuda()
bias = torch.randn(4).cuda()

# Call the function
output = leaky_relu_conv2d(input, weight, bias, stride=2, padding=1, dilation=1, groups=1, negative_slope=0.01, inplace=False)

# Print the output shape
print(output.shape)  # Expected: (2, 4, 5, 5)
