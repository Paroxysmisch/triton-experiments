import torch

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> torch.Tensor:
    # Handle dim=None by reducing all dimensions
    if dim is None:
        dim = tuple(range(input.dim()))
    
    # Compute the standard deviation using PyTorch's built-in function
    result = torch.std(input, dim=dim, correction=correction, keepdim=keepdim)
    
    # If an output tensor is provided, copy the result into it
    if out is not None:
        out.copy_(result)
        return out
    return result
