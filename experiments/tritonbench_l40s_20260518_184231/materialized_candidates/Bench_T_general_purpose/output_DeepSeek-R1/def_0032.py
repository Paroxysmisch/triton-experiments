import torch
from typing import Tuple, Optional

def linalg_eig(A: torch.Tensor, *, out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(A):
        raise TypeError(f"Expected tensor input, got {type(A)}")
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(f"Expected square matrix (batch dimensions allowed), got shape {A.shape}")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    
    # Synchronize if on CUDA
    if A.is_cuda:
        torch.cuda.synchronize(A.device)
    
    # Handle output tuple
    if out is not None:
        if len(out) != 2:
            raise ValueError("out must be a tuple of two tensors")
        out_val, out_vec = out
        if not (torch.is_tensor(out_val) and torch.is_tensor(out_vec)):
            raise TypeError("out tuple must contain tensors")
        out_val.resize_as_(eigenvalues).copy_(eigenvalues)
        out_vec.resize_as_(eigenvectors).copy_(eigenvectors)
        return (out_val, out_vec)
    else:
        return (eigenvalues, eigenvectors)
