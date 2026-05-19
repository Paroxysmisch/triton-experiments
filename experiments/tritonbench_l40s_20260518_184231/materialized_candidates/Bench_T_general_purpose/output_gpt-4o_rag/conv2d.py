import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, kH, kW,
    strideH, strideW, padH, padW, dilationH, dilationW, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    """
    Triton kernel for 2D convolution.
    """
    # Compute the output dimensions
    oH = (iH + 2 * padH - dilationH * (kH - 1) - 1) // strideH + 1
    oW = (iW + 2 * padW - dilationW * (kW - 1) - 1) // strideW + 1

    # Obtain program ids for parallelization
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    out_y = tl.program_id(2) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_x = tl.program_id(3) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Ensure we are within the output bounds
    out_y = tl.where(out_y < oH, out_y, 0)
    out_x = tl.where(out_x < oW, out_x, 0)

    # Initialize the output value
    output_val = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over the input channels
    for c in range(in_channels // groups):
        for ky in range(kH):
            for kx in range(kW):
                # Calculate input indices
                in_y = out_y * strideH + ky * dilationH - padH
                in_x = out_x * strideW + kx * dilationW - padW

                # Load input values
                input_val = tl.load(input_ptr + batch_id * in_channels * iH * iW + c * iH * iW + in_y * iW + in_x, mask=(in_y >= 0) & (in_y < iH) & (in_x >= 0) & (in_x < iW), other=0.0)

                # Load weight values
                weight_val = tl.load(weight_ptr + out_channel_id * (in_channels // groups) * kH * kW + c * kH * kW + ky * kW + kx)

                # Accumulate the convolution result
                output_val += input_val * weight_val

    # Add bias if provided
    if bias_ptr:
        bias_val = tl.load(bias_ptr + out_channel_id)
        output_val += bias_val

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * oH * oW + out_channel_id * oH * oW + out_y * oW + out_x, output_val)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Wrapper function for 2D convolution using Triton.
    """
    # Ensure input is on CUDA
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA device."

    # Handle default parameters
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Get input and weight dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    oW = (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    # Prepare output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)

    # Configure grid size
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    grid = (batch_size, out_channels, (oH + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H, (oW + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)

    # Launch Triton kernel
    conv2d_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        iH=iH, iW=iW, kH=kH, kW=kW,
        strideH=stride[0], strideW=stride[1],
        padH=padding[0], padW=padding[1],
        dilationH=dilation[0], dilationW=dilation[1],
        groups=groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return output
