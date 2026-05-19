def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Check if other is a tensor or a scalar
    if other is not None and not torch.is_tensor(other):
        other = torch.tensor(other)

    # Apply the convolution
    output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # Add the other tensor or scalar to the convolution result
    if other is not None:
        output += alpha * other

    return output
