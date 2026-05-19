import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union

def gelu_conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = 'none',
    out: Optional[Tensor] = None
) -> Tensor:
    # Compute the 2D convolution
    conv_output = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # Apply GELU activation
    gelu_output = F.gelu(conv_output, approximate=approximate)
    
    # Handle the output tensor if provided
    if out is not None:
        out.copy_(gelu_output)
        return out
    return gelu_output
