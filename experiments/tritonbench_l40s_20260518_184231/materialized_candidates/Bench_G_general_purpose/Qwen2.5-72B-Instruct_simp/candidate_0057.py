import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    input_ptr,  # Pointer to the input tensor
    weight_ptr, # Pointer to the weight tensor
    output_ptr, # Pointer to the output tensor
    input_shape, # (N, C_in, H_in, W_in)
    weight_shape, # (C_out, C_in // groups, kernel_H, kernel_W)
    output_shape, # (N, C_out, H_out, W_out)
    stride, # (stride_H, stride_W)
    padding, # (padding_H, padding_W)
    dilation, # (dilation_H, dilation_W)
    groups, # Number of groups
    input_strides, # (stride_N, stride_C_in, stride_H_in, stride_W_in)
    weight_strides, # (stride_C_out, stride_C_in, stride_kernel_H, stride_kernel_W)
    output_strides, # (stride_N, stride_C_out, stride_H_out, stride_W_out)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_KH: tl.constexpr,
    BLOCK_SIZE_KW: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_c_out = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_w = tl.program_id(axis=3)

    n_start = pid_n * BLOCK_SIZE_N
    c_out_start = pid_c_out * BLOCK_SIZE_C_OUT
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    for n in range(n_start, n_start + BLOCK_SIZE_N):
        if n >= input_shape[0]:
            break
        for c_out in range(c_out_start, c_out_start + BLOCK_SIZE_C_OUT):
            if c_out >= output_shape[1]:
                break
            for h in range(h_start, h_start + BLOCK_SIZE_H):
                if h >= output_shape[2]:
                    break
                for w in range(w_start, w_start + BLOCK_SIZE_W):
                    if w >= output_shape[3]:
                        break
                    out_idx = n * output_strides[0] + c_out * output_strides[1] + h * output_strides[2] + w * output_strides[3]
                    output_ptr += out_idx
                    acc = 0.0
                    for c_in_group in range(weight_shape[1]):
                        for kh in range(weight_shape[2]):
                            for kw in range(weight_shape[3]):
                                input_h = h * stride[0] - padding[0] + kh * dilation[0]
                                input_w = w * stride[1] - padding[1] + kw * dilation[1]
                                if 0 <= input_h < input_shape[2] and 0 <= input_w < input_shape[3]:
                                    input_idx = n * input_strides[0] + (c_out // groups * groups + c_in_group) * input_strides[1] + input_h * input_strides[2] + input_w * input_strides[3]
                                    weight_idx = c_out * weight_strides[0] + c_in_group * weight_strides[1] + kh * weight_strides[2] + kw * weight_strides[3]
                                    input_val = tl.load(input_ptr + input_idx)
                                    weight_val = tl.load(weight_ptr + weight_idx)
                                    acc += input_val * weight_val
                    tl.store(output_ptr, acc)

import torch
import triton
import triton.language as tl

def conv2d_forward(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Ensure input and weight are on the same device
    assert input.device == weight.device, "Input and weight must be on the same device"
    device = input.device

    # Get input and weight dimensions
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, kernel_H, kernel_W = weight.shape

    # Calculate output dimensions
    H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_H - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_W - 1) - 1) // stride[1] + 1

    # Initialize output tensor
    output = torch.empty((N, C_out, H_out, W_out), device=device, dtype=input.dtype)

    # Define block sizes
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_C_OUT = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C_IN = 16
    BLOCK_SIZE_KH = 16
    BLOCK_SIZE_KW = 16

    # Define grid dimensions
    grid = (
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT,
        (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )

    # Define strides
    input_strides = (C_in * H_in * W_in, H_in * W_in, W_in, 1)
    weight_strides = (C_in_per_group * kernel_H * kernel_W, kernel_H * kernel_W, kernel_W, 1)
    output_strides = (C_out * H_out * W_out, H_out * W_out, W_out, 1)

    # Launch the kernel
    conv2d_forward_kernel[grid](
        input, weight, output,
        input.shape, weight.shape, output.shape,
        stride, padding, dilation, groups,
        input_strides, weight_strides, output_strides,
        BLOCK_SIZE_N, BLOCK_SIZE_C_OUT, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_IN, BLOCK_SIZE_KH, BLOCK_SIZE_KW
    )

    return output
