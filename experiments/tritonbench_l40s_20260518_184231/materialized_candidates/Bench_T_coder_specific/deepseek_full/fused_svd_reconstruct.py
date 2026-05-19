import torch
import triton
import triton.language as tl

@triton.jit
def fused_svd_reconstruct(A: tl.tensor) -> tl.tensor:
    # Compute SVD
    m, n = A.shape
    U, S, Vh = torch.linalg.svd(A)
    # Reconstruct A
    A_reconstructed = U @ torch.diag(S) @ Vh
    return A_reconstructed

def test_fused_svd_reconstruct() -> None:
    # Create random input
    A = torch.randn(size=(10, 5), dtype=torch.float32, device="cuda")
    # Call function
    A_reconstructed = fused_svd_reconstruct(A)
    # Verify if the reconstructed matrix is close to the original one
    assert torch.allclose(A, A_reconstructed)

if __name__ == "__main__":
    test_fused_svd_reconstruct()
