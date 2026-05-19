import triton
import triton.language as tl
import torch

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    in_channels, out_channels, groups,
    iH, iW, kH, kW, oH, oW,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the position of the block in the output
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    # Calculate the starting index for the output
    out_index = (batch_id * out_channels + out_ch_id) * oH * oW + out_h * oW + out_w

    # Initialize the output value
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Calculate the convolution
    for g in range(groups):
        for ic in range(in_channels // groups):
            for kh in range(kH):
                for kw in range(kW):
                    in_h = out_h * stride_h - pad_h + kh * dilation_h
                    in_w = out_w * stride_w - pad_w + kw * dilation_w
                    if 0 <= in_h < iH and 0 <= in_w < iW:
                        in_index = ((batch_id * in_channels + g * (in_channels // groups) + ic) * iH + in_h) * iW + in_w
                        weight_index = ((out_ch_id * (in_channels // groups) + ic) * kH + kh) * kW + kw
                        acc += tl.load(input_ptr + in_index) * tl.load(weight_ptr + weight_index)

    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_ch_id)

    # Store the result
    tl.store(output_ptr + out_index, acc)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Extract dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Handle stride, padding, and dilation
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

    # Calculate output dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid = (batch_size, out_channels, oH, oW)
    conv2d_kernel[grid](
        input, weight, bias, output,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        in_channels, out_channels, groups,
        iH, iW, kH, kW, oH, oW,
        BLOCK_SIZE=1  # You can adjust block size for optimization
    )

    return output
