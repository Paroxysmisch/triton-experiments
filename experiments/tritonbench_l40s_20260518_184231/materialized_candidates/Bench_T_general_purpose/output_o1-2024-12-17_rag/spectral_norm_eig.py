import triton
import triton.language as tl
import torch

def spectral_norm_eig(A, *, out=None):
    """
    Computes the spectral norm of a square matrix (or batch of square matrices) A
    using its eigenvalues. The spectral norm is the largest absolute value among
    the eigenvalues of A.
    
    A (Tensor): Shape (*, n, n), where * is zero or more batch dimensions.
    out (Tensor, optional): If provided, must have shape (*). Ignored if None.
                            Default: None.
    Returns:
        Tensor of shape (*) containing the spectral norms for each matrix in A.
    """
    # Compute eigenvalues using PyTorch
    # For real A, returns real or complex eigenvalues
    # For complex A, returns complex eigenvalues
    eigenvals = torch.linalg.eigvals(A)
    abs_eigenvals = eigenvals.abs()

    # Largest absolute eigenvalue per matrix in the batch
    # The last dimension has size n (the number of eigenvalues)
    spectral_vals, _ = abs_eigenvals.max(dim=-1)

    # If out is specified, copy the result into out, else return a new tensor
    if out is None:
        return spectral_vals
    else:
        out.copy_(spectral_vals)
        return out
