import torch

def solve_symmetric_ldl(A, b, *, hermitian=False, out=None):
    """
    Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition.
    
    Parameters:
    A (Tensor): A symmetric (or Hermitian) matrix of shape (*, n, n).
    b (Tensor): Right-hand side tensor of shape (*, n) or (*, n, k).
    hermitian (bool, optional): Whether to consider A as Hermitian. Default: False.
    out (Tensor, optional): Output tensor. Ignored if None. Default: None.
    
    Returns:
    Tensor: Solution to the linear system.
    """
    # Ensure A is a square matrix
    assert A.dim() >= 2 and A.size(-1) == A.size(-2), "A must be a square matrix"
    
    # Perform LDL decomposition
    L, D, _ = torch.linalg.ldl_factor(A, hermitian=hermitian)
    
    # Reconstruct A using L and D
    if hermitian:
        A_reconstructed = L @ D @ L.transpose(-2, -1).conj()
    else:
        A_reconstructed = L @ D @ L.transpose(-2, -1)
    
    # Solve the linear system
    solution = torch.linalg.solve(A_reconstructed, b, out=out)
    
    return solution

# Example usage:
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
b = torch.tensor([1.0, 2.0])
x = solve_symmetric_ldl(A, b)
print(x)
