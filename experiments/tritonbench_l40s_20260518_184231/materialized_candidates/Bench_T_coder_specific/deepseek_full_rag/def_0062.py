import triton
import torch

@triton.jit
def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Compute y += alpha * x
    y += alpha * x
    # Compute dot product of the modified y with itself
    dot_product = torch.dot(y, y)
    return dot_product

# Example usage
y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
x = torch.tensor([4.0, 5.0, 6.0], device='cuda')
alpha = 2.0
result = scaled_add_dot(y, x, alpha)
print(result)
