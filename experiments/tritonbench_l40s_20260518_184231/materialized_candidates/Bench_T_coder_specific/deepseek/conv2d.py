import triton
import torch

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    input = torch.from_numpy(input)
    weight = torch.from_numpy(weight)
    if bias is not None:
        bias = torch.from_numpy(bias)

    output = triton.ops.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )

    return output.numpy()
