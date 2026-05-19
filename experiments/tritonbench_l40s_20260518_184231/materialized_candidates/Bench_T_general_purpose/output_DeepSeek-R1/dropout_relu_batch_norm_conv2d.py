import torch
import torch.nn.functional as F

def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False
) -> torch.Tensor:
    # Apply Conv2d
    x = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Compute mean and variance for batch normalization using current batch statistics
    if x.dim() != 4:
        raise ValueError("Expected 4D input (N, C, H, W)")
    num_channels = x.size(1)
    # Using current batch stats, no tracked running stats
    x = F.batch_norm(x, None, None, training=training, momentum=0.0, eps=1e-5)
    
    # Apply ReLU
    x = F.relu(x, inplace=inplace)
    
    # Apply Dropout
    x = F.dropout(x, p=p, training=training, inplace=inplace)
    
    return x
