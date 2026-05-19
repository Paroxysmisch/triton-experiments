import torch

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    return torch.rsqrt(torch.tensordot(a, b, dims=dims))
