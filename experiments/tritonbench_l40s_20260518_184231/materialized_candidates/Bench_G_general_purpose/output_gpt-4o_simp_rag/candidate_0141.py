import torch

# Example matrices
a = torch.randn(128, 64, device='cuda', dtype=torch.float32)
b = torch.randint(-128, 127, (64, 128), device='cuda', dtype=torch.int8)
b_scale = torch.rand(128, device='cuda', dtype=torch.float32)

# Perform matrix multiplication with dequantization
c = matmul_dequantize_int8(a, b, b_scale)

print(c)
