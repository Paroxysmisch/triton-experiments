import torch

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    return torch.softmax(input, dim)
