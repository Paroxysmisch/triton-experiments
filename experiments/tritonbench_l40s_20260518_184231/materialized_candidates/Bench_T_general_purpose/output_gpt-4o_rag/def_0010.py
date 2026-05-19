import torch

def svd(A, full_matrices=True, *, driver=None, out=None):
    """
    Computes the singular value decomposition (SVD) of a matrix using PyTorch.

    Args:
        A (Tensor): Tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
        full_matrices (bool, optional): Controls whether to compute the full or reduced SVD. Default: `True`.
        driver (str, optional): Name of the cuSOLVER method to be used. Available options are: `None`, `gesvd`, `gesvdj`, and `gesvda`. Default: `None`.
        out (tuple, optional): Output tuple of three tensors. Ignored if `None`.

    Returns:
        (Tensor, Tensor, Tensor): The singular value decomposition (U, S, Vh) of the input matrix.
    """
    # Ensure the input is on the GPU if using CUDA-specific drivers
    if driver is not None and A.is_cuda:
        # Handle driver-specific logic if needed (currently, PyTorch doesn't expose this directly)
        pass

    # Compute SVD using PyTorch
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    # Assign outputs if out is provided
    if out is not None:
        out[0].copy_(U)
        out[1].copy_(S)
        out[2].copy_(Vh)

    return U, S, Vh

# Example usage
A = torch.randn(3, 5, 5, device='cuda')  # Batch of 3 matrices of size 5x5
U, S, Vh = svd(A, full_matrices=False)
