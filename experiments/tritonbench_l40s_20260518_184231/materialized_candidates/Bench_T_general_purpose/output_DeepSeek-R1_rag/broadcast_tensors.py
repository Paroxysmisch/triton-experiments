import torch
from typing import List

def broadcast_tensors(*tensors) -> List[torch.Tensor]:
    # Check all tensors are on the same device and have compatible types
    if len(tensors) == 0:
        return []
    
    device = tensors[0].device
    dtype = tensors[0].dtype
    
    for t in tensors:
        if t.device != device:
            raise RuntimeError("All tensors must be on the same device")
        if t.dtype != dtype:
            raise RuntimeError("All tensors must have the same dtype")
    
    # Use PyTorch's built-in broadcast tensors which handles view-based expansion
    return torch.broadcast_tensors(*tensors)
