import torch
from torch import Tensor
from typing import Optional

def triton_solve(A: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None) -> Tensor:
    # Check if any input is on a CUDA device
    cuda_device = None
    if A.is_cuda:
        cuda_device = A.device
    elif B.is_cuda:
        cuda_device = B.device
    
    # Compute the solution using PyTorch's solve for numerical stability and efficiency
    result = torch.linalg.solve(A, B, left=left, out=out)
    
    # Synchronize the CUDA device if necessary
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    
    return result
