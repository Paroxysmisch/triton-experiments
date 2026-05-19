import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_height, input_width, input_channels,
    output_height, output_width, output_channels,
    kernel_height, kernel_width, stride_height, stride_width,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_o, weight_stride_i, weight_stride_h, weight_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
):
    # Define the block indices
    n_idx = tl.program_id(0)
    c_out_idx = tl.program_id(1)
    h_out_idx = tl.program_id(2)
    w_out_idx = tl.program_id(3)

    # Compute the starting point of the block
    n_start = n_idx * BLOCK_SIZE_N
    c_out_start = c_out_idx * BLOCK_SIZE_C
    h_out_start = h_out_idx * BLOCK_SIZE_H
    w_out_start = w_out_idx * BLOCK_SIZE_W

    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)

    # Iterate over the input channels and kernel dimensions
    for c_in in range(input_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                # Compute input indices
                h_in = h_out_start * stride_height + kh
                w_in = w_out_start * stride_width + kw

                # Load input and weight values
                input_val = tl.load(input_ptr + n_start * input_stride_n + c_in * input_stride_c + h_in * input_stride_h + w_in * input_stride_w)
                weight_val = tl.load(weight_ptr + c_out_start * weight_stride_o + c_in * weight_stride_i + kh * weight_stride_h + kw * weight_stride_w)

                # Perform the convolution
                acc += input_val * weight_val

    # Store the result
    tl.store(output_ptr + n_start * output_stride_n + c_out_start * output_stride_c + h_out_start * output_stride_h + w_out_start * output_stride_w, acc)

def conv2d_forward(input_tensor, weight_tensor, stride=(1, 1)):
    # Extract dimensions
    n, c_in, h_in, w_in = input_tensor.shape
    c_out, _, kh, kw = weight_tensor.shape

    # Calculate output dimensions
    h_out = (h_in - kh) // stride[0] + 1
    w_out = (w_in - kw) // stride[1] + 1

    # Prepare output tensor
    output_tensor = torch.empty((n, c_out, h_out, w_out), device=input_tensor.device, dtype=input_tensor.dtype)

    # Define block sizes
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8

    # Launch the Triton kernel
    grid = (triton.cdiv(n, BLOCK_SIZE_N), triton.cdiv(c_out, BLOCK_SIZE_C), triton.cdiv(h_out, BLOCK_SIZE_H), triton.cdiv(w_out, BLOCK_SIZE_W))
    conv2d_forward_kernel[grid](
        input_tensor, weight_tensor, output_tensor,
        h_in, w_in, c_in,
        h_out, w_out, c_out,
        kh, kw, stride[0], stride[1],
        *input_tensor.stride(), *weight_tensor.stride(), *output_tensor.stride(),
        BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W
    )

    return output_tensor
