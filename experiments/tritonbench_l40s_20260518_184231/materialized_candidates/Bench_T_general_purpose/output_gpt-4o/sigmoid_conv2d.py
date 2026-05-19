import triton
import triton.language as tl
import torch

@triton.jit
def sigmoid_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels,
    iH, iW, kH, kW, oH, oW,
    stride_h, stride_w, pad_h, pad_w,
    dilation_h, dilation_w, groups,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the position in the output tensor
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    col_idx = tl.program_id(3)

    # Initialize output value
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the kernel dimensions
    for kh in range(kH):
        for kw in range(kW):
            for in_channel_idx in range(in_channels // groups):
                # Calculate input positions
                row_pos = row_idx * stride_h - pad_h + kh * dilation_h
                col_pos = col_idx * stride_w - pad_w + kw * dilation_w

                # Check bounds
                if (0 <= row_pos < iH) and (0 <= col_pos < iW):
                    # Load input and weight values
                    input_offset = (
                        batch_idx * in_channels * iH * iW +
                        (out_channel_idx // groups) * in_channels * iH * iW +
                        in_channel_idx * iH * iW +
                        row_pos * iW + col_pos
                    )
                    weight_offset = (
                        out_channel_idx * (in_channels // groups) * kH * kW +
                        in_channel_idx * kH * kW +
                        kh * kW + kw
                    )
                    input_val = tl.load(input_ptr + input_offset)
                    weight_val = tl.load(weight_ptr + weight_offset)

                    # Accumulate the convolution result
                    acc += input_val * weight_val

    # Add bias if provided
    if bias_ptr:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val

    # Apply sigmoid activation function
    acc = 1 / (1 + tl.exp(-acc))

    # Store the result
    output_offset = (
        batch_idx * out_channels * oH * oW +
        out_channel_idx * oH * oW +
        row_idx * oW + col_idx
    )
    tl.store(output_ptr + output_offset, acc)


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation, dilation
    else:
        dilation_h, dilation_w = dilation

    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Prepare output tensor
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Launch Triton kernel
    grid = (batch_size, out_channels, oH, oW)
    sigmoid_conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        iH=iH, iW=iW, kH=kH, kW=kW, oH=oH, oW=oW,
        stride_h=stride_h, stride_w=stride_w,
        pad_h=pad_h, pad_w=pad_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups,
        BLOCK_SIZE=1  # Adjust this as necessary
    )

    return out
