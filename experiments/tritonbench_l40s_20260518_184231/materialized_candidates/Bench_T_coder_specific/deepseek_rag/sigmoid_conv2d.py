import torch
import torch.nn.functional as F

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    """
    Applies a 2D convolution with specified filters and applies the sigmoid activation function.

    Parameters:
    -----------
    input : torch.Tensor
        The input tensor of shape `(minibatch, in_channels, iH, iW)`.
    weight : torch.Tensor
        The convolution filters of shape `(out_channels, in_channels / groups, kH, kW)`.
    bias : torch.Tensor, optional
        Optional bias tensor of shape `(out_channels)`. Default: None.
    stride : int or tuple, optional
        The stride of the convolution kernel. Can be a single number or a tuple `(sH, sW)`. Default: 1.
    padding : int, tuple, or string, optional
        Padding on both sides of the input. Can be 'valid', 'same', single number, or tuple `(padH, padW)`. Default: 0.
    dilation : int or tuple, optional
        The spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of groups to split the input into. Default: 1.
    out : torch.Tensor, optional
        The output tensor.

    Returns:
    --------
    torch.Tensor
        A tensor of the same shape as the input tensor, containing the result of the 2D convolution and sigmoid activation.
    """

    # Apply 2D convolution
    output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # Apply sigmoid activation
    output = torch.sigmoid(output)

    return output
