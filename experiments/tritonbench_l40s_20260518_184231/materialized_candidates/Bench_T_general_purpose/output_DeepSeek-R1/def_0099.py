import torch
import torch.nn.functional as F

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    # Apply GELU activation
    activated = F.gelu(input, approximate=approximate)
    # Compute standard deviation with correction
    result = torch.std(activated, dim=dim, keepdim=keepdim, correction=correction, out=out)
    return result
