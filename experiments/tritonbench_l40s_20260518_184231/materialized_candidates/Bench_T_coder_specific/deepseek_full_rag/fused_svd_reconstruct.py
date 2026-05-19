import torch
import triton
import triton.language as tl

@triton.jit
def fused_svd_reconstruct(A: torch.Tensor):
    # SVD of A: A = U Σ V^H
    U, S, Vh = torch.linalg.svd(A)
    # Reconstruct A as A_reconstructed = U Σ V^H
    A_reconstructed = torch.matmul(U, torch.matmul(torch.diag(S), Vh))
    return A_reconstructed

def fused_svd_reconstruct_wrapper(A: torch.Tensor) -> torch.Tensor:
    return fused_svd_reconstruct(A)
