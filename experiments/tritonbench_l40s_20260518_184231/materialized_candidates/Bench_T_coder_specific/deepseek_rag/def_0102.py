import torch

def softmax_mul(input, other, dim, dtype=None, out=None):
    softmaxed = torch.softmax(input, dim=dim, dtype=dtype)
    if isinstance(other, torch.Tensor):
        return torch.mul(softmaxed, other, out=out)
    elif isinstance(other, (int, float)):
        return torch.mul(softmaxed, other, out=out)
    else:
        raise TypeError("other must be a Tensor or a Number")
