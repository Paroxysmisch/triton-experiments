import torch
import torch.nn.functional as F

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False):
    if bias is None:
        bias = torch.zeros(weight.size(0), device=input.device)

    linear_out = F.linear(input, weight, bias)
    sigmoid_out = F.sigmoid(linear_out)

    if training:
        sigmoid_out = F.dropout(sigmoid_out, p=p, training=True, inplace=inplace)

    return sigmoid_out
