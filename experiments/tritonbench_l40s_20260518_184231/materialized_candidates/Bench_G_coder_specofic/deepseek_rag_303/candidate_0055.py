import triton
import torch

# Example usage
a = torch.randn(2, 1024, device='cuda')
b = torch.randn(2, 1024, device='cuda')
a, b, c = swiglu_forward(a, b)
dc = torch.randn_like(c)
a, b = swiglu_backward(a, b, dc)
