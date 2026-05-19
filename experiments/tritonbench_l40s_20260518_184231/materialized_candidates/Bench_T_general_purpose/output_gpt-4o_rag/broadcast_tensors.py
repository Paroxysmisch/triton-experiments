import torch
from typing import List

def broadcast_tensors(*tensors) -> List[torch.Tensor]:
    """
    Broadcasts the given tensors according to broadcasting semantics.
    
    Args:
        *tensors: Any number of tensors of the same type.
        
    Returns:
        A list of tensors that have been broadcasted to the same shape.
    """
    # Use PyTorch's built-in function to broadcast tensors
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    
    # Check for in-place operation risks and clone if necessary
    for i, tensor in enumerate(broadcasted_tensors):
        if tensor.storage().size() != tensor.numel():
            # More than one element refers to the same memory location
            # Clone the tensor to avoid in-place operation issues
            broadcasted_tensors[i] = tensor.clone()
    
    return broadcasted_tensors

# Example usage
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
print(a.size())  # Output: torch.Size([2, 3])
print(a)         # Output: tensor([[0, 1, 2], [0, 1, 2]])
