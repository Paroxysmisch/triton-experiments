import torch

def ifftshift(input, dim=None):
    if dim is None:
        dim = tuple(range(input.ndim))
    return torch.fft.ifftshift(input, dim=dim)
