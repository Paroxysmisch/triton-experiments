import torch
import torch.nn.functional as F

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Apply 2D convolution
    output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply ReLU activation
    output = F.relu(output, inplace=inplace)
    
    return output
