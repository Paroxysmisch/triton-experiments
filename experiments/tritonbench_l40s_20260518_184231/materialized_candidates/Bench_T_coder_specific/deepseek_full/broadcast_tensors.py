import torch

def broadcast_tensors(*tensors):
    # Ensure all tensors are on the same device
    device = tensors[0].device
    tensors = [tensor.to(device) for tensor in tensors]

    # Calculate the maximum shape by finding the shape with the maximum size in each dimension
    max_shape = [max(tensor.size(i) for tensor in tensors) for i in range(tensors[0].dim())]

    # Expand each tensor to the maximum shape in its dimension
    tensors = [tensor.expand(max_shape) for tensor in tensors]

    return tensors

# Test case
x = torch.arange(3).view(1, 3)
y = torch.arange(2).view(2, 1)
a, b = broadcast_tensors(x, y)
assert a.size() == torch.Size([2, 3])
assert torch.all(a == torch.tensor([[0, 1, 2],[0, 1, 2]]))
