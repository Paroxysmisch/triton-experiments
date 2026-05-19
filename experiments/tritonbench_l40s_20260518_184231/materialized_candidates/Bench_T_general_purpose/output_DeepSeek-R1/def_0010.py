import torch
from typing import Optional, Tuple

def svd(A: torch.Tensor, full_matrices: bool = True, *, driver: Optional[str] = None, out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Validate driver argument
    if driver is not None:
        if A.device.type != 'cuda':
            raise ValueError("The 'driver' argument is only supported on CUDA inputs.")
        valid_drivers = ['gesvd', 'gesvdj', 'gesvda']
        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver '{driver}'. Expected one of: {valid_drivers} or None.")
    
    # Perform SVD using PyTorch's built-in function (driver selection not directly supported in PyTorch's public API)
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    
    # Handle the output tuple if provided
    if out is not None:
        if len(out) != 3:
            raise ValueError("The 'out' tuple must contain exactly three tensors.")
        out[0].copy_(U)
        out[1].copy_(S)
        out[2].copy_(Vh)
        return (out[0], out[1], out[2])
    else:
        return (U, S, Vh)
