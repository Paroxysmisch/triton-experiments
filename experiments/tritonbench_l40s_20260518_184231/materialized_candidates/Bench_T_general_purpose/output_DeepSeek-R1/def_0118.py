import torch

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve the upper triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    # Add the scaled vector alpha * y to the solution x
    x += alpha * y
    return x
