import torch

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Solves the triangular system of linear equations Ax = b, where A is an upper triangular matrix.
    Then, adds a scaled version of the vector y to the solution x.
    
    Parameters:
    A (Tensor): A triangular matrix of shape (n, n).
    b (Tensor): Right-hand side vector or matrix of shape (n,) or (n, k).
    y (Tensor): Vector to be scaled and added, must have shape (n,) or broadcastable to (n,).
    alpha (float): Scaling factor for the vector y.
    
    Returns:
    Tensor: The solution x with the scaled vector y added.
    """
    # Solve the triangular system Ax = b
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add the scaled vector alpha * y to the solution x
    x += alpha * y
    
    return x
