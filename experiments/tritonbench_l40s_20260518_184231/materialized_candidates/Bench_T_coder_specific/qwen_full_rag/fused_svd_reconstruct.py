import torch
import triton
import triton.language as tl
from typing import Optional
from ..utils.shape_utils import _contiguous
from .svd import _svd

# Wrapper function to reconstruct matrix A using its Singular Value Decomposition (SVD)
def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Ensure the input matrix is contiguous
    A = _contiguous(A)
    # Check if input is a batched matrix or a single matrix
    is_batched = A.dim() > 2
    if is_batched:
        # Reshape input to a 2D tensor if it's batched
        A = A.view((-1,) + A.shape[-2:])
    
    M, N = A.shape[-2:]
    # Perform SVD decomposition on the input matrix
    U, S, Vh = _svd(A, compute_uv=True, full_matrices=False)
    # Construct the diagonal matrix Sigma from the singular values
    Sigma = tl.zeros((M, N), dtype=U.dtype).to(U.device)
    r = min(M, N)
    Sigma[:r, :r] = S
    
    # Reconstruct the matrix A using U, Sigma, and Vh
    A_reconstructed = U @ Sigma @ Vh
    
    if is_batched:
        # Reshape the reconstructed matrix back to its original batched shape
        A_reconstructed = A_reconstructed.view(A.shape)
    
    return A_reconstructed
