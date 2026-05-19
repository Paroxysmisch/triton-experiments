# triton_conv2d.py

import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                  stride_h, stride_w, pad_h, pad_w, 
                  dilation_h, dilation_w, 
                  in_channels, out_channels, 
                  kH, kW, 
                  n_batch, iH, iW, 
                  group_count, 
                  BLOCK_SIZE: tl.constexpr):
    # Define the block and grid sizes
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    row_idx = tl.arange(0, BLOCK_SIZE)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Calculate output dimensions
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    # Load input and weight
    input_data = tl.load(input_ptr + (batch_idx, :, row_idx[:, None] * stride_h + col_idx[None, :] * stride_w))
    weight_data = tl.load(weight_ptr + (out_channel_idx, :, :, :))

    # Perform convolution
    output_data = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(kH):
        for l in range(kW):
            output_data += input_data * weight_data[:, :, k, l]

    # Add bias if provided
    if bias_ptr is not None:
        bias_data = tl.load(bias_ptr + out_channel_idx)
        output_data += bias_data

    # Store the result
    tl.store(output_ptr + (batch_idx, out_channel_idx, row_idx, col_idx), output_data)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Validate input shapes
    n_batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Handle stride and padding
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    # Allocate output tensor
    oH = (iH + 2 * pad_h - (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - (kW - 1) - 1) // stride_w + 1
    output = torch.empty((n_batch, out_channels, oH, oW), dtype=input.dtype)

    # Launch the kernel
    grid = (n_batch, out_channels, oH, oW)
    conv2d_kernel[grid](input, weight, bias, output, 
                        stride_h, stride_w, pad_h, pad_w, 
                        dilation_h, dilation_w, 
                        in_channels, out_channels, 
                        kH, kW, 
                        n_batch, iH, iW, 
                        groups)

    return output
