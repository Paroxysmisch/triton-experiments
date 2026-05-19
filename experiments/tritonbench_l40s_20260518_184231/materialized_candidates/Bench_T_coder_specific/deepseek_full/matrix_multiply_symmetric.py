import torch

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    C = alpha * torch.mm(A, B) + beta * C
    C = alpha * torch.mm(C, C.T) + beta * C
    return C
