import torch
from torch import nn
import triton

class DropoutReluBatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, p=0.5, training=True, inplace=False):
        super(DropoutReluBatchNormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p)
        self.inplace = inplace

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.relu(x, inplace=self.inplace)
        x = self.dropout(x)
        return x
