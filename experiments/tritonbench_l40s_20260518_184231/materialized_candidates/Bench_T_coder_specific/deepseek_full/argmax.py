import torch

def argmax(input, dim=None, keepdim=False):
    return torch.max(input, dim, keepdim).indices
