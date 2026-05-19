import torch

def spectral_norm_eig(A, *, out=None):
    """
    Computes the spectral norm of a square matrix using its eigenvalues.
    
    Parameters:
    A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of square matrices.
    out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
    
    Returns:
    Tensor: The spectral norm of each matrix in the batch.
    """
    # Check if A is a square matrix
    if A.size(-1) != A.size(-2):
        raise ValueError("Input must be a square matrix")

    # Use PyTorch's eigenvalue function for demonstration
    # Triton doesn't have direct support for eigenvalue computation
    # Eigenvalues can be complex, so we take the absolute values
    eigenvalues = torch.linalg.eigvals(A)
    spectral_norms = eigenvalues.abs().max(dim=-1).values

    # If an output tensor is provided, store the result there
    if out is not None:
        out.copy_(spectral_norms)
        return out

    return spectral_norms

# Example usage
A = torch.rand(3, 4, 4, dtype=torch.cfloat)  # A batch of 3 complex 4x4 matrices
result = spectral_norm_eig(A)
print(result)
