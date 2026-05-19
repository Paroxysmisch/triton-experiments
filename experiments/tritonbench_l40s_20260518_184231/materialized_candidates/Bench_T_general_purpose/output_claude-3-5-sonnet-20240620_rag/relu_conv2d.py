import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                  iH, iW, oH, oW, in_channels, out_channels, 
                  kH, kW, stride, padding, dilation, groups, 
                  BLOCK_SIZE: tl.constexpr):
    # Calculate the output position
    batch_id = tl.program_id(0)
    row = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Initialize output tensor
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the kernel
    for g in range(groups):
        for k in range(out_channels // groups):
            for h in range(kH):
                for w in range(kW):
                    # Calculate input indices
                    h_in = row + h * stride - padding
                    w_in = col + w * stride - padding
                    mask = (h_in >= 0) & (h_in < iH) & (w_in >= 0) & (w_in < iW)
                    input_val = tl.load(input_ptr + batch_id * in_channels * iH * iW + 
                                         g * (in_channels // groups) * iH * iW + 
                                         h_in * iW + w_in, mask=mask, other=0.0)
                    weight_val = tl.load(weight_ptr + g * (out_channels // groups) * in_channels * kH * kW + 
                                          k * in_channels * kH * kW + h * kW + w)
                    output += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + batch_id * out_channels + g * (out_channels // groups) + k)
        output += bias_val

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * oH * oW + row * oW + col, output)

@triton.jit
def relu_kernel(output_ptr, oH, oW, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    output_val = tl.load(output_ptr + row * oW + col)
    output_val = tl.maximum(output_val, 0)  # Apply ReLU
    tl.store(output_ptr + row * oW + col, output_val)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    """
    Applies a 2D convolution followed by ReLU activation.

    Parameters:
    -----------
    input : torch.Tensor
        Input tensor of shape (minibatch, in_channels, iH, iW).
    weight : torch.Tensor
        Convolution filters of shape (out_channels, in_channels / groups, kH, kW).
    bias : torch.Tensor, optional
        Optional bias tensor of shape (out_channels). Default: None.
    stride : int or tuple, optional
        Stride of the convolution kernel. Default: 1.
    padding : int, tuple, or string, optional
        Padding added to all sides of the input. Default: 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.
    inplace : bool, optional
        If True, will perform ReLU operation in-place. Default: False.

    Returns:
    --------
    torch.Tensor
        A tensor resulting from the 2D convolution followed by ReLU activation.
    """

    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."
    if bias is not None:
        assert bias.device == device, "Bias must be on the same CUDA device."

    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    oH = (iH + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    oW = (iW + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    output = torch.empty((minibatch, out_channels, oH, oW), device=device)

    BLOCK_SIZE = 16  # Adjust as needed
    grid = (minibatch, (oH + BLOCK_SIZE - 1) // BLOCK_SIZE, (oW + BLOCK_SIZE - 1) // BLOCK_SIZE)

    # Call the convolution kernel
    conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        iH=iH,
        iW=iW,
        oH=oH,
        oW=oW,
        in_channels=in_channels,
        out_channels=out_channels,
        kH=kH,
        kW=kW,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Call the ReLU kernel
    relu_kernel[grid](
        output_ptr=output,
        oH=oH,
        oW=oW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
