import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, batch_size, iH, iW, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """
    Applies a 2D convolution to the input tensor using Triton.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    weight_ptr : tl.tensor
        Pointer to the weight tensor in global memory.
    bias_ptr : tl.tensor
        Pointer to the bias tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    batch_size : int
        Number of samples in the batch.
    iH : int
        Height of the input tensor.
    iW : int
        Width of the input tensor.
    kH : int
        Height of the kernel.
    kW : int
        Width of the kernel.
    stride_h : int
        Stride in the height dimension.
    stride_w : int
        Stride in the width dimension.
    pad_h : int
        Padding in the height dimension.
    pad_w : int
        Padding in the width dimension.
    dilation_h : int
        Dilation in the height dimension.
    dilation_w : int
        Dilation in the width dimension.
    groups : int
        Number of groups.
    BLOCK_SIZE_N : tl.constexpr
        Block size for the batch dimension.
    BLOCK_SIZE_H : tl.constexpr
        Block size for the height dimension.
    BLOCK_SIZE_W : tl.constexpr
        Block size for the width dimension.
    """

    pid_n = tl.program_id(0)  # Batch index
    pid_h = tl.program_id(1)  # Height index
    pid_w = tl.program_id(2)  # Width index

    # Compute the output indices
    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Compute the output shape
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Mask for valid indices
    mask_n = n < batch_size
    mask_h = h < oH
    mask_w = w < oW
    mask = mask_n[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Initialize the output tensor
    output = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels // groups), dtype=tl.float32)

    # Load the bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr, mask=tl.arange(0, out_channels // groups) < (out_channels // groups))

    # Iterate over the output channels
    for oc in range(out_channels // groups):
        # Iterate over the input channels
        for ic in range(in_channels // groups):
            # Iterate over the kernel
            for kh in range(kH):
                for kw in range(kW):
                    # Compute the input indices
                    input_h = h * stride_h - pad_h + kh * dilation_h
                    input_w = w * stride_w - pad_w + kw * dilation_w

                    # Mask for valid input indices
                    input_mask = (input_h >= 0) & (input_h < iH) & (input_w >= 0) & (input_w < iW)

                    # Load the input and weight
                    input_val = tl.load(input_ptr + (n * in_channels * iH * iW + ic * iH * iW + input_h * iW + input_w), mask=input_mask, other=0.0)
                    weight_val = tl.load(weight_ptr + (oc * in_channels * kH * kW + ic * kH * kW + kh * kW + kw))

                    # Perform the convolution
                    output[:, :, :, oc] += input_val * weight_val

    # Add the bias if provided
    if bias_ptr is not None:
        output += bias[None, None, None, :]

    # Store the output
    tl.store(output_ptr + (n * out_channels * oH * oW + oc * oH * oW + h * oW + w), output, mask=mask)
