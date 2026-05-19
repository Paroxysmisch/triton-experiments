import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot(y, x, alpha):
    # Compute y += alpha * x
    tl.store(y + tl.arange(0, 1), tl.load(y + tl.arange(0, 1)) + alpha * tl.load(x + tl.arange(0, 1)))
    # Compute the dot product of y with itself
    dot_product = tl.sum(tl.load(y + tl.arange(0, 1)) * tl.load(y + tl.arange(0, 1)))
    return dot_product.to(tl.float32)

# Example usage
y = torch.tensor([1.0, 2.0, 3.0], device='cuda')
x = torch.tensor([4.0, 5.0, 6.0], device='cuda')
alpha = 2.0
result = scaled_add_dot(y, x, alpha)
print(result)
