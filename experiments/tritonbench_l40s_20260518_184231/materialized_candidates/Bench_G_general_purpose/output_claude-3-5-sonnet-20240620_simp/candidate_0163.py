import torch

# Example usage
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float32).contiguous()
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
c = matmul(a, b, use_leaky_relu=True, leaky_relu_alpha=0.1)
