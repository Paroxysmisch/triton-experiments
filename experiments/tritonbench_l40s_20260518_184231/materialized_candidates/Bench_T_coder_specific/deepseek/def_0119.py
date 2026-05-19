import torch
import torch.nn.functional as F

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Convolution operation
    output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # Pixel shuffle upscaling
    output = F.pixel_shuffle(output, upscale_factor)

    return output
