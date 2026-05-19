|system|>import torch

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Perform the matrix-matrix product and update C
    C[:] = alpha * torch.mm(A, B) + beta * C
    # Update C to be the weighted sum of its own product with its transpose
    C[:] = alpha * torch.mm(C, C.T) + beta * C
    return C
