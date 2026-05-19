import torch
import torch.linalg

def det(A: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    sign, logabsdet = torch.linalg.slogdet(A)
    result = sign * torch.exp(logabsdet)
    if out is not None:
        out.copy_(result)
        return out
    return result
