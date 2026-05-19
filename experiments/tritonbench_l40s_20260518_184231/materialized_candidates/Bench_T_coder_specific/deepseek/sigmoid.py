import torch

def sigmoid(input, *, out=None):
    return torch.special.expit(input, out=out)
