import torch

def spectral_norm_eig(A, *, out=None):
    """
    Computes the spectral norm of a batch of square matrices using their eigenvalues.
    
    Parameters:
    A (Tensor): Tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
    out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.
    
    Returns:
    Tensor: The spectral norm for each matrix in the batch.
    """
    # Ensure the input is a square matrix
    assert A.dim() >= 2 and A.size(-1) == A.size(-2), "Input must be a batch of square matrices"
    
    # Compute eigenvalues using PyTorch
    eigenvalues = torch.linalg.eigvals(A)
    
    # Compute the spectral norm (largest absolute eigenvalue)
    spectral_norms = torch.max(eigenvalues.abs(), dim=-1).values
    
    # If an output tensor is provided, use it
    if out is not None:
        out.copy_(spectral_norms)
        return out
    
    return spectral_norms

# Example usage
A = torch.randn(3, 4, 4, dtype=torch.cfloat)  # A batch of 3 complex 4x4 matrices
spectral_norms = spectral_norm_eig(A)
print(spectral_norms)
