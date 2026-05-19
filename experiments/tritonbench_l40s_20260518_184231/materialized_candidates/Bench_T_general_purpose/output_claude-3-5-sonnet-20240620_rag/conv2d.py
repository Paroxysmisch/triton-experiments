import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr, 
    stride_h, stride_w, pad_h, pad_w, 
    dilation_h, dilation_w, 
    in_channels, out_channels, 
    iH, iW, oH, oW, 
    groups, BLOCK_SIZE: tl.constexpr
):
    """
    Applies a 2D convolution to the input tensor using Triton for GPU acceleration.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    weight_ptr : tl.tensor
        Pointer to the weight tensor (filters) used for convolution.
    bias_ptr : tl.tensor
        Pointer to the bias tensor (optional).
    stride_h, stride_w : int
        The stride of the convolving kernel in height and width.
    pad_h, pad_w : int
        Implicit paddings on both sides of the input in height and width.
    dilation_h, dilation_w : int
        The spacing between kernel elements in height and width.
    in_channels, out_channels : int
        Number of input and output channels.
    iH, iW : int
        Height and width of the input tensor.
    oH, oW : int
        Height and width of the output tensor.
    groups : int
        Number of groups for grouped convolution.
    BLOCK_SIZE : tl.constexpr
        Size of the block used in the Triton kernel grid.

    Notes:
    ------
    This function operates on 2D input data, performing convolution over a specified kernel size.
    Handles boundaries by zero-padding the input when necessary.
    """
    
    batch_id = tl.program_id(0)
    h, w = tl.program_id(1), tl.program_id(2)
    
    # Calculate the input coordinates
    input_h = h * stride_h - pad_h
    input_w = w * stride_w - pad_w
    
    # Initialize the output value
    output_val = tl.zeros((out_channels,), dtype=tl.float32)

    for c in range(in_channels // groups):
        for kh in range(weight_ptr.shape[2]):  # kernel height
            for kw in range(weight_ptr.shape[3]):  # kernel width
                # Calculate the input index
                input_idx_h = input_h + kh * dilation_h
                input_idx_w = input_w + kw * dilation_w
                
                # Load input value if within bounds
                if 0 <= input_idx_h < iH and 0 <= input_idx_w < iW:
                    input_val = tl.load(input_ptr + batch_id * in_channels * iH * iW + c * iH * iW + input_idx_h * iW + input_idx_w)
                else:
                    input_val = 0.0
                
                # Load weight value
                weight_val = tl.load(weight_ptr + c * weight_ptr.shape[1] * weight_ptr.shape[2] * weight_ptr.shape[3] + kh * weight_ptr.shape[3] + kw)
                
                # Accumulate the result
                output_val += input_val * weight_val

    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + batch_id)
        output_val += bias_val

    # Store the result
    tl.store(output_ptr + batch_id * out_channels * oH * oW + h * oW + w, output_val)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Performs a 2D convolution on an input tensor using the specified parameters.

    Parameters:
    -----------
    input : torch.Tensor
        A 4D tensor of shape (minibatch, in_channels, iH, iW).
    weight : torch.Tensor
        A 4D tensor of shape (out_channels, in_channels/groups, kH, kW).
    bias : torch.Tensor, optional
        A 1D tensor of shape (out_channels). Default: None.
    stride : int or tuple, optional
        The stride of the convolving kernel. Default: 1.
    padding : int or str, optional
        Implicit paddings on both sides of the input. Default: 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default: 1.
    groups : int, optional
        Split input into groups. Default: 1.

    Returns:
    --------
    torch.Tensor
        A 4D tensor containing the convolved output.

    Notes:
    ------
    The function divides the computation into blocks using Triton, performing convolution efficiently
    on GPU by parallelizing over the batch dimension and the spatial dimensions.
    """

    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."
    if bias is not None:
        assert bias.device == device, "Bias must be on the same CUDA device."

    B, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    # Calculate output dimensions
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else (padding[0], padding[1])
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    oH = (iH + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    output = torch.empty((B, out_channels, oH, oW), device=device)

    BLOCK_SIZE = 256
    grid = (B, oH, oW)

    conv2d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        weight_ptr=weight,
        bias_ptr=bias,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        in_channels=in_channels,
        out_channels=out_channels,
        iH=iH,
        iW=iW,
        oH=oH,
        oW=oW,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
