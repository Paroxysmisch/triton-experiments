import torch

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Solves the triangular system Ax = b and adds alpha * y to the solution x.

    Parameters:
    A (Tensor): A triangular matrix of shape `(n, n)`.
    b (Tensor): Right-hand side vector or matrix of shape `(n,)` or `(n, k)`.
    y (Tensor): Vector to be scaled and added, must have shape `(n,)` or broadcastable to `(n,)`.
    alpha (float): Scaling factor for the vector y.

    Returns:
    Tensor: The solution vector x after adding the scaled vector.
    """
    # Solve the triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add the scaled vector alpha * y to the solution x
    x += alpha * y

    return x

# Example usage
A = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device='cuda')  # Upper triangular matrix
b = torch.tensor([3.0, 6.0], device='cuda')  # Right-hand side vector
y = torch.tensor([1.0, 2.0], device='cuda')  # Vector to be scaled and added
alpha = 0.5  # Scaling factor

result = solve_and_add_scaled_vector(A, b, y, alpha)
print(result)
