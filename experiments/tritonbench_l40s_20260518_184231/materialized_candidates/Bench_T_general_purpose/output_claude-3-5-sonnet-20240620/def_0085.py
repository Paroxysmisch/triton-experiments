import torch
from torch import Tensor
from torch.linalg import eig, inv, diag

def matrix_power_eig(A: Tensor, k: float, *, out: Tensor = None) -> Tensor:
    # Ensure A is a square matrix
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError("Input tensor A must be a square matrix.")

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)

    # Compute the diagonal matrix of eigenvalues raised to the power k
    eigenvalues_powered = eigenvalues ** k
    D_k = diag(eigenvalues_powered)

    # Compute the matrix power A^k using the formula A^k = V diag(Λ^k) V^(-1)
    A_k = eigenvectors @ D_k @ inv(eigenvectors)

    # Handle output tensor
    if out is not None:
        out.copy_(A_k)
        return out
    return A_k
