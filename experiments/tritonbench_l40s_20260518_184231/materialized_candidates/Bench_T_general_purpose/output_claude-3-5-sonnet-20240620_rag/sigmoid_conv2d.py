import torch
import triton
import triton.language as tl

@triton.jit
def conv2d(input_ptr, weight_ptr, bias_ptr, out_ptr, 
            minibatch, in_channels, iH, iW, 
            out_channels, kH, kW, 
            stride_h, stride_w, 
            pad_h, pad_w, 
            dilation_h, dilation_w, 
            groups, BLOCK_SIZE: tl.constexpr):
    # Implementation of 2D convolution
    # ... existing code ...

@triton.jit
def sigmoid_conv2d(input_ptr, weight_ptr, bias_ptr, out_ptr, 
                   minibatch, in_channels, iH, iW, 
                   out_channels, kH, kW, 
                   stride=1, padding=0, dilation=1, 
                   groups=1, BLOCK_SIZE: tl.constexpr):
    """
    Applies a 2D convolution followed by a sigmoid activation function.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    weight_ptr : tl.tensor
        Pointer to the convolution filters.
    bias_ptr : tl.tensor
        Pointer to the optional bias tensor.
    out_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    minibatch : int
        Number of input samples.
    in_channels : int
        Number of input channels.
    iH : int
        Input height.
    iW : int
        Input width.
    out_channels : int
        Number of output channels.
    kH : int
        Kernel height.
    kW : int
        Kernel width.
    stride : int or tuple
        Stride of the convolution kernel.
    padding : int or tuple
        Padding on both sides of the input.
    dilation : int or tuple
        Dilation of the kernel.
    groups : int
        Number of groups for grouped convolution.
    BLOCK_SIZE : tl.constexpr
        Size of the block used in the Triton kernel grid.
    """
    
    # Call the convolution kernel
    conv2d(input_ptr, weight_ptr, bias_ptr, out_ptr, 
            minibatch, in_channels, iH, iW, 
            out_channels, kH, kW, 
            stride_h=stride, stride_w=stride, 
            pad_h=padding, pad_w=padding, 
            dilation_h=dilation, dilation_w=dilation, 
            groups=groups, BLOCK_SIZE=BLOCK_SIZE)

    # Apply sigmoid activation
    idx = tl.program_id(0)
    result = tl.load(out_ptr + idx)
    sigmoid_result = 1 / (1 + tl.exp(-result))
    tl.store(out_ptr + idx, sigmoid_result)

def sigmoid_conv2d_wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    """
    Wrapper function for the sigmoid_conv2d Triton kernel.

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
        Padding on both sides of the input. Default: 0.
    dilation : int or tuple, optional
        Dilation of the kernel. Default: 1.
    groups : int, optional
        Number of groups for grouped convolution. Default: 1.
    out : torch.Tensor, optional
        Output tensor. Default: None.

    Returns:
    --------
    torch.Tensor
        Output tensor after applying convolution and sigmoid activation.
    """
    
    device = input.device
    assert device.type == "cuda", "Input tensor must be on a CUDA device for Triton ops."
    assert input.device == weight.device, "Input and weight must be on the same CUDA device."
    
    if bias is not None:
        assert bias.device == device, "Bias must be on the same CUDA device."

    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape

    if out is None:
        out = torch.empty((minibatch, out_channels, (iH + 2 * padding - kH) // stride + 1, (iW + 2 * padding - kW) // stride + 1), device=device)

    BLOCK_SIZE = 256
    grid = (minibatch * out.shape[2] * out.shape[3],)
    
    # Call the Triton kernel
    sigmoid_conv2d[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        minibatch=minibatch,
        in_channels=in_channels,
        iH=iH,
        iW=iW,
        out_channels=out_channels,
        kH=kH,
        kW=kW,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
