import torch

def generate_uniform_numbers(n, from_, to):
    return torch.rand(n) * (to - from_) + from_
