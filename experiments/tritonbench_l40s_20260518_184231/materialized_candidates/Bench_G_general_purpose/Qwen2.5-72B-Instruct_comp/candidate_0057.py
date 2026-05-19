import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_shape, weight_shape, output_shape,
    input_strides, weight_strides, output_strides,
    kernel_size, stride, padding, groups,
    BLOCK_BATCH, BLOCK_IN_FEATURES, BLOCK_OUT_FEATURES,
    FP16, TF32
):
    # Block indices
    pid_batch = tl.program_id(axis=0)
    pid_out_h = tl.program_id(axis=1)
    pid_out_w = tl.program_id(axis=2)
    pid_out_c = tl.program_id(axis=3)

    # Compute the output indices
    batch_start = pid_batch * BLOCK_BATCH
    out_h_start = pid_out_h * BLOCK_OUT_FEATURES
    out_w_start = pid_out_w * BLOCK_OUT_FEATURES
    out_c_start = pid_out_c * BLOCK_OUT_FEATURES

    # Compute the input and weight indices
    in_h_start = (out_h_start * stride) - padding
    in_w_start = (out_w_start * stride) - padding

    # Load the input and weight tensors
    input = tl.load(input_ptr + input_strides[0] * batch_start + input_strides[1] * in_h_start + input_strides[2] * in_w_start + input_strides[3] * out_c_start)
    weight = tl.load(weight_ptr + weight_strides[0] * out_c_start + weight_strides[1] * in_h_start + weight_strides[2] * in_w_start + weight_strides[3] * out_c_start)

    # Initialize the output tensor
    output = tl.zeros((BLOCK_BATCH, BLOCK_OUT_FEATURES, BLOCK_OUT_FEATURES, BLOCK_OUT_FEATURES), dtype=tl.float32)

    # Compute the convolution
    for h in range(kernel_size):
        for w in range(kernel_size):
            for c in range(BLOCK_IN_FEATURES):
                input_idx = input_strides[0] * batch_start + input_strides[1] * (in_h_start + h) + input_strides[2] * (in_w_start + w) + input_strides[3] * c
                weight_idx = weight_strides[0] * out_c_start + weight_strides[1] * h + weight_strides[2] * w + weight_strides[3] * c
                output += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)

    # Store the output tensor
    output_idx = output_strides[0] * batch_start + output_strides[1] * out_h_start + output_strides[2] * out_w_start + output_strides[3] * out_c_start
    tl.store(output_ptr + output_idx, output)

import torch
import triton
import triton.language as tl

def conv2d_forward(input, weight, kernel_size, stride, padding, groups, FP16=False, TF32=False):
    # Get input and weight dimensions
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels_per_group, k_height, k_width = weight.shape

    # Compute output dimensions
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    # Initialize output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=input.device, dtype=input.dtype)

    # Compute block and grid sizes
    BLOCK_BATCH = 16
    BLOCK_IN_FEATURES = 16
    BLOCK_OUT_FEATURES = 16

    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (out_height + BLOCK_OUT_FEATURES - 1) // BLOCK_OUT_FEATURES,
        (out_width + BLOCK_OUT_FEATURES - 1) // BLOCK_OUT_FEATURES,
        (out_channels + BLOCK_OUT_FEATURES - 1) // BLOCK_OUT_FEATURES
    )

    # Launch the kernel
    conv2d_forward_kernel[grid](
        input, weight, output,
        input.shape, weight.shape, output.shape,
        input.stride(), weight.stride(), output.stride(),
        kernel_size, stride, padding, groups,
        BLOCK_BATCH, BLOCK_IN_FEATURES, BLOCK_OUT_FEATURES,
        FP16, TF32
    )

    return output
