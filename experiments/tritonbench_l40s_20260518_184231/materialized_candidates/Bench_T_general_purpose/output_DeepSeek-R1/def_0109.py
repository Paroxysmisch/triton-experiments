import torch
from torch import Tensor

def pseudoinverse_svd(A: Tensor, *, full_matrices: bool = True, rcond: float = 1e-15, out: Tensor = None) -> Tensor:
    # Compute the SVD of the input tensor A
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Compute the threshold for singular values
    S_max = S[..., 0:1]  # Keep dimensions for broadcasting
    threshold = rcond * S_max
    
    # Invert the singular values above the threshold, set others to zero
    S_plus = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
    
    # Get the number of singular values (k = min(m, n))
    k = S.size(-1)
    
    # Handle the case for full_matrices=True
    if full_matrices:
        # Determine the dimensions of the original matrix
        m = A.size(-2)
        n = A.size(-1)
        # Create a zero matrix of shape (..., n, m) to hold Sigma_plus
        Sigma_plus = torch.zeros(*S_plus.shape[:-1], n, m, dtype=S_plus.dtype, device=S_plus.device)
        # Fill the diagonal with the inverted singular values up to k elements
        diag_indices = torch.arange(k, device=S_plus.device)
        Sigma_plus[..., diag_indices, diag_indices] = S_plus
    else:
        # Create a diagonal matrix from S_plus for reduced SVD case
        Sigma_plus = torch.diag_embed(S_plus)
    
    # Compute the conjugate transpose of Vh and U
    Vh_mH = Vh.mH
    U_mH = U.mH
    
    # Calculate the pseudoinverse: V^H @ Sigma_plus @ U^H
    if full_matrices:
        # When full_matrices=True, perform the matrix multiplication explicitly
        pinv = Vh_mH @ (Sigma_plus @ U_mH)
    else:
        # Optimized computation for reduced SVD using broadcasting
        scaled_Vh = Vh_mH * S_plus.unsqueeze(-2)
        pinv = scaled_Vh @ U_mH
    
    # Handle the output tensor if provided
    if out is not None:
        if not out.is_complex() and pinv.is_complex():
            raise RuntimeError("Can't cast complex output to real tensor")
        out.copy_(pinv)
        return out
    else:
        return pinv
