import triton
import torch

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
        weight = weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)

    output = torch.nn.functional.linear(input, weight, bias)
    output = triton.ops.log_softmax(output, dim)

    return output
