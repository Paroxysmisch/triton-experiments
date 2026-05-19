import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale(A, B, alpha, beta):
    B = alpha * torch.mm(torch.tril(A), B)
    C = beta * B
    return C

# Example Usage
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
alpha = 0.5
beta = 0.5

output = tril_mm_and_scale(A, B, alpha, beta)
