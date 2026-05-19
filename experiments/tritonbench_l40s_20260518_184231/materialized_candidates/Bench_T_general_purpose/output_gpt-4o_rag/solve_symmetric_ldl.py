import torch

def solve_symmetric_ldl(A, b, *, hermitian=False, out=None):
    """
    Solves a symmetric (or Hermitian) linear system A x = b using LDL decomposition.
    
    Parameters:
    A (Tensor): A symmetric (or Hermitian) matrix of shape (*, n, n).
    b (Tensor): A tensor of shape (*, n) or (*, n, k) representing the right-hand side.
    hermitian (bool, optional): Whether to treat A as Hermitian. Default: False.
    out (Tensor, optional): Output tensor. If None, ignored. Default: None.
    
    Returns:
    Tensor: The solution to the linear system.
    """
    # Perform LDL decomposition
    L, D, perm = torch.linalg.ldl_factor(A, hermitian=hermitian)
    
    # Reconstruct the matrix A using L and D
    if hermitian:
        A_reconstructed = L @ D @ L.transpose(-2, -1).conj()
    else:
        A_reconstructed = L @ D @ L.transpose(-2, -1)
    
    # Solve the linear system using the reconstructed A
    x = torch.linalg.solve(A_reconstructed, b)
    
    # If an output tensor is provided, copy the result to it
    if out is not None:
        out.copy_(x)
        return out
    
    return x

# Example usage
A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
b = torch.tensor([1.0, 2.0])
x = solve_symmetric_ldl(A, b)
print(x)
