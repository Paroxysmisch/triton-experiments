import torch

def dot_product(x, y):
    # Ensure x and y are 1D tensors
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Both input tensors must be 1-dimensional")
    
    if x.size(0) != y.size(0):
        raise ValueError("Input tensors must be of the same size")

    return torch.dot(x, y)
