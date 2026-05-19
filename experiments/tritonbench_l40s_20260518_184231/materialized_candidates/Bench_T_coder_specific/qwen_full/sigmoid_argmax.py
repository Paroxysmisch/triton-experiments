import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_argmax(input, dim=None, keepdim=False):
    input = torch.sigmoid(input)
    return torch.argmax(input, dim=dim, keepdim=keepdim)

def wrapper(input, dim=None, keepdim=False):
    return sigmoid_argmax(input, dim, keepdim)
