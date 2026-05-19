import torch

def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b, L)
