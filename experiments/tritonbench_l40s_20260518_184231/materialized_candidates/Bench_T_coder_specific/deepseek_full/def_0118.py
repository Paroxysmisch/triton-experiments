import torch
import triton

@triton.jit
def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve the triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    # Add the scaled vector y to x
    x += alpha * y
    return x
