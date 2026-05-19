import torch

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Perform tensor contraction using torch.tensordot
    tensordot_result = torch.tensordot(a, b, dims)
    
    # Apply the reciprocal square root element-wise
    output = torch.rsqrt(tensordot_result)
    
    return output
