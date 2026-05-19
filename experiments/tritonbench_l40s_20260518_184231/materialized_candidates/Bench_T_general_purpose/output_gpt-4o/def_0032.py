import torch

def linalg_eig(A, *, out=None):
    """
    Computes the eigenvalue decomposition of a square matrix if it exists.
    
    Args:
        A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions consisting of diagonalizable matrices.
        out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.
        
    Returns:
        (Tensor, Tensor): A tuple containing the eigenvalues and eigenvectors.
    """
    # Ensure input is a square matrix or batch of square matrices
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Input must be a square matrix or a batch of square matrices")
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Synchronize the device with the CPU if on CUDA
    if A.is_cuda:
        torch.cuda.synchronize()
    
    # If out is provided, store the results in the provided tensors
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise ValueError("out must be a tuple of two tensors")
        out[0].copy_(eigenvalues)
        out[1].copy_(eigenvectors)
        return out
    
    return eigenvalues, eigenvectors
