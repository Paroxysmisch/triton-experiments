import torch
from torch.nn import BatchNorm2d
from torch import sigmoid

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    # Create BatchNorm2d layer
    bn = BatchNorm2d(input.size(1), momentum=momentum, eps=eps)
    bn.running_mean = running_mean
    bn.running_var = running_var
    if weight is not None and bias is not None:
        bn.weight = weight
        bn.bias = bias
    bn.train(training)
    
    # Apply BatchNorm and SiLU
    return bn(input) * sigmoid(input)
