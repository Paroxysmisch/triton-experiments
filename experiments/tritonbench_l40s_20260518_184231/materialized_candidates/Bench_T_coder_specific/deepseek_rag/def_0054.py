import torch
import torch.nn as nn
import torch.nn.functional as F

class GELUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, approximate='none'):
        super(GELUConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.approximate = approximate

    def forward(self, x):
        x = self.conv(x)
        if self.approximate == 'none':
            x = F.gelu(x)
        elif self.approximate == 'tanh':
            x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
        else:
            raise ValueError("Invalid approximation method for GELU. Must be 'none' or 'tanh'.")
        return x
