import torch

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    assert input.dim() == 4, "Input tensor should be of shape (minibatch, in_channels, iH, iW)"
    assert weight.dim() == 4, "Weight tensor should be of shape (out_channels, in_channels/groups, kH, kW)"
    assert bias is None or bias.dim() == 1, "Bias tensor should be of shape (out_channels)"
    assert isinstance(upscale_factor, int) and upscale_factor > 0, "Upscale factor should be a positive integer"

    # Continue with the rest of your function...
