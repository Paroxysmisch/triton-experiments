import triton
import triton.language as tl

@triton.jit
def conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, groups,
    input_shape, weight_shape, output_shape,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Extract the dimensions
    B, C, H, W = input_shape
    K, C_group, R, S = weight_shape
    _, K_out, H_out, W_out = output_shape

    # Compute the grid and block indices
    pid_n = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    # Compute the block bounds
    n_start = pid_n * BLOCK_SIZE_N
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    n_end = min(n_start + BLOCK_SIZE_N, B)
    h_end = min(h_start + BLOCK_SIZE_H, H_out)
    w_end = min(w_start + BLOCK_SIZE_W, W_out)

    # Initialize the output block
    output_block = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over the input and weight blocks
    for n in range(n_start, n_end):
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                # Initialize the output value for this position
                output_value = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

                # Compute the input and weight indices
                for r in range(R):
                    for s in range(S):
                        for c in range(C // groups):
                            input_r = h * stride[0] + r * dilation[0] - padding[0]
                            input_s = w * stride[1] + s * dilation[1] - padding[1]
                            if 0 <= input_r < H and 0 <= input_s < W:
                                input_idx = n * C * H * W + c * H * W + input_r * W + input_s
                                weight_idx = (n % K) * (C // groups) * R * S + c * R * S + r * S + s
                                input_value = tl.load(input_ptr + input_idx)
                                weight_value = tl.load(weight_ptr + weight_idx)
                                output_value += input_value * weight_value

                # Apply the bias if provided
                if bias_ptr is not None:
                    bias_idx = n % K
                    bias_value = tl.load(bias_ptr + bias_idx)
                    output_value += bias_value

                # Apply the sigmoid activation
                output_value = 1 / (1 + tl.exp(-output_value))

                # Store the output value
                output_idx = n * K_out * H_out * W_out + (n % K) * H_out * W_out + h * W_out + w
                tl.store(output_ptr + output_idx, output_value)

# Define the grid and block sizes
BLOCK_SIZE_N = 16
BLOCK_SIZE_H = 16
BLOCK_SIZE_W = 16
BLOCK_SIZE_C = 16
BLOCK_SIZE_K = 16

import torch
import triton
import triton.language as tl

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Ensure input and weight are on the same device
    device = input.device
    input = input.to(device)
    weight = weight.to(device)
    if bias is not None:
        bias = bias.to(device)

    # Ensure the stride, padding, and dilation are tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Extract the input and weight shapes
    B, C, H, W = input.shape
    K, C_group, R, S = weight.shape

    # Compute the output shape
    H_out = (H + 2 * padding[0] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    output_shape = (B, K, H_out, W_out)

    # Allocate the output tensor
    if out is None:
        out = torch.empty(output_shape, device=device, dtype=input.dtype)

    # Define the grid and block sizes
    grid = (B, H_out, W_out)
    block = (BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W)

    # Launch the Triton kernel
    conv2d_sigmoid_kernel[grid, block](
        input, weight, bias, out,
        stride, padding, dilation, groups,
        (B, C, H, W), (K, C_group, R, S), output_shape,
        BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C, BLOCK_SIZE_K
    )

    return out
