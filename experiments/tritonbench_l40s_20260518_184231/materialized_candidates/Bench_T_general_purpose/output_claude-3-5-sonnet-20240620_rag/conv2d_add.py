import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr, 
                  minibatch, in_channels, iH, iW, out_channels, kH, kW, 
                  stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, 
                  groups, alpha, BLOCK_SIZE: tl.constexpr):
    # Calculate the output dimensions
    out_h = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    out_w = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1

    # Get the program ID
    batch_id = tl.program_id(0)
    h_id = tl.program_id(1)
    w_id = tl.program_id(2)

    # Calculate the output index
    output_index = (batch_id * out_channels * out_h * out_w) + (h_id * out_w) + w_id

    # Initialize the output value
    output_value = tl.zeros((), dtype=tl.float32)

    # Iterate over the kernel
    for kh in range(kH):
        for kw in range(kW):
            # Calculate the input coordinates
            h_in = h_id * stride_h - pad_h + kh * dilation_h
            w_in = w_id * stride_w - pad_w + kw * dilation_w

            # Check if the input coordinates are within bounds
            if 0 <= h_in < iH and 0 <= w_in < iW:
                for g in range(groups):
                    input_index = (batch_id * in_channels * iH * iW) + \
                                   (g * (in_channels // groups) * iH * iW) + \
                                   (h_in * iW) + w_in
                    weight_index = (g * (out_channels // groups) * kH * kW) + \
                                   (kh * kW) + kw
                    output_value += tl.load(input_ptr + input_index) * tl.load(weight_ptr + weight_index)

    # Add bias if provided
    if bias_ptr is not None:
        output_value += tl.load(bias_ptr + (batch_id * out_channels))

    # Add the other tensor or scalar scaled by alpha
    if other_ptr is not None:
        output_value += alpha * tl.load(other_ptr + output_index)

    # Store the result
    tl.store(output_ptr + output_index, output_value)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, 
                dilation=1, groups=1, alpha=1, out=None):
    """
    Applies a 2D convolution over an input image using specified filters and an optional bias,
    then adds another tensor or scalar to the convolution result, scaled by alpha.

    Parameters:
    -----------
    input : torch.Tensor
        The input tensor of shape (minibatch, in_channels, iH, iW).
    weight : torch.Tensor
        The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
    bias : torch.Tensor, optional
        Optional bias tensor of shape (out_channels). Default: None.
    other : torch.Tensor or Number, optional
        The tensor or number to add to the convolution result. Default: None.
    stride : int or tuple, optional
        The stride of the convolution kernel. Default: 1.
    padding : int, tuple, or string, optional
        Padding on both sides of the input. Default: 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of groups to split the input into. Default: 1.
    alpha : Number, optional
        The multiplier for other. Default: 1.
    out : torch.Tensor, optional
        The output tensor.

    Returns:
    --------
    torch.Tensor
        The output tensor after applying the convolution and addition.
    """

    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."
    if bias is not None:
        assert bias.device == device, "Bias must be on the same CUDA device."
    if other is not None:
        assert other.device == device, "Other must be on the same CUDA device."

    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Handle padding
    if isinstance(padding, str):
        if padding == 'same':
            pad_h = (iH % stride + (stride - 1)) // 2
            pad_w = (iW % stride + (stride - 1)) // 2
        elif padding == 'valid':
            pad_h = pad_w = 0
    else:
        pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding

    # Create output tensor
    if out is None:
        out_h = (iH + 2 * pad_h - (dilation * (kH - 1) + 1)) // stride + 1
        out_w = (iW + 2 * pad_w - (dilation * (kW - 1) + 1)) // stride + 1
        output = torch.empty((minibatch, out_channels, out_h, out_w), device=device)
    else:
        output = out

    BLOCK_SIZE = 256
    grid = (minibatch, (output.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE, (output.shape[3] + BLOCK_SIZE - 1) // BLOCK_SIZE)

    conv2d_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        other_ptr=other,
        output_ptr=output,
        minibatch=minibatch,
        in_channels=in_channels,
        iH=iH,
        iW=iW,
        out_channels=out_channels,
        kH=kH,
        kW=kW,
        stride_h=stride if isinstance(stride, int) else stride[0],
        stride_w=stride if isinstance(stride, int) else stride[1],
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation if isinstance(dilation, int) else dilation[0],
        dilation_w=dilation if isinstance(dilation, int) else dilation[1],
        groups=groups,
        alpha=alpha,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
