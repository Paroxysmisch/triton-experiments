import triton
import triton.language as tl
import torch
import torch.nn.functional as F

def dropout(x, p, train=True):
    if not train:
        return x
    
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    
    # Create a mask tensor with the same shape as x
    x_keep = torch.rand_like(x, dtype=torch.float32)
    
    # Apply dropout
    if p > 0:
        x_keep = (x_keep < (1 - p)).float()
        x = x * x_keep / (1 - p)
    
    return x
