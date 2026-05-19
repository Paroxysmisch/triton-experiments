import torch
import torch.nn.functional as F

def mul_relu(input, other, inplace=False, out=None):
    # Perform element-wise multiplication
    if inplace:
        input.mul_(other)
    else:
        input = torch.mul(input, other)

    # Apply ReLU
    output = F.relu(input, inplace=inplace, out=out)

    return output
