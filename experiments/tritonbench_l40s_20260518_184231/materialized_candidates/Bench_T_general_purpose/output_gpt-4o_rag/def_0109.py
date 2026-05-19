import torch
import triton
import triton.language as tl

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.

    Args:
        A (Tensor): Input tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        full_matrices (bool, optional): If `True`, compute the full SVD. If `False`, compute the reduced SVD.
        rcond (float, optional): Relative condition number threshold.
        out (Tensor, optional): Output tensor. Ignored if `None`.

    Returns:
        Tensor: The pseudoinverse of the input matrix.
    """
    # Check input dimensions
    assert A.dim() >= 2, "Input tensor must have at least 2 dimensions"
    batch_dims = A.shape[:-2]
    m, n = A.shape[-2:]

    # Perform SVD using PyTorch
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    # Invert singular values with thresholding
    max_singular_value = S.max(dim=-1, keepdim=True).values
    cutoff = rcond * max_singular_value
    S_inv = torch.where(S > cutoff, 1.0 / S, torch.tensor(0.0, dtype=S.dtype, device=S.device))

    # Reconstruct pseudoinverse
    V = Vh.conj().transpose(-2, -1)
    U = U.conj().transpose(-2, -1)
    S_inv_mat = torch.diag_embed(S_inv)

    pseudoinv = V @ S_inv_mat @ U

    if out is not None:
        out.copy_(pseudoinv)
        return out
    return pseudoinv

# Example usage
A = torch.randn(3, 4, 5, dtype=torch.float32)  # Example batch of matrices
pseudoinv_A = pseudoinverse_svd(A)
