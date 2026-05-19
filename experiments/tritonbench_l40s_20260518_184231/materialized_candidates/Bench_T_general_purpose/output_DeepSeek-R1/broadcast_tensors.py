import torch
from typing import List

def broadcast_tensors(*tensors) -> List[torch.Tensor]:
    # Compute the broadcastable shape
    if not tensors:
        return []
    
    # Align all shapes to the same number of dimensions by padding with 1s on the left
    shapes = [tensor.shape for tensor in tensors]
    max_ndim = max(len(shape) for shape in shapes)
    aligned_shapes = []
    for shape in shapes:
        aligned = (1,) * (max_ndim - len(shape)) + shape
        aligned_shapes.append(aligned)
    
    # Compute the broadcasted shape
    broadcasted_shape = []
    for dims in zip(*aligned_shapes):
        max_dim = max(dims)
        if any(d != 1 and d != max_dim for d in dims):
            raise ValueError("Shapes cannot be broadcasted together")
        broadcasted_shape.append(max_dim)
    broadcasted_shape = tuple(broadcasted_shape)
    
    # Expand each tensor to the broadcasted shape
    broadcasted_tensors = []
    for tensor in tensors:
        # Add leading singleton dimensions if needed
        tensor_exp = tensor
        while tensor_exp.dim() < len(broadcasted_shape):
            tensor_exp = tensor_exp.unsqueeze(0)
        # Expand to the broadcasted shape
        tensor_exp = tensor_exp.expand(broadcasted_shape)
        broadcasted_tensors.append(tensor_exp)
    
    return broadcasted_tensors
