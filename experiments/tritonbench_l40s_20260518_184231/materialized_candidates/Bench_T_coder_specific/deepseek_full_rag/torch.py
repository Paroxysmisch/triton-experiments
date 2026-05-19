import torch

def torch.permute_copy(input, *dims):
    """
    Wrapper for torch.permute that ensures all output tensors are freshly created instead of aliasing the input.
    """
    return input.permute(*dims).clone()
