import triton
import torch

@triton.jit
def fused_svd_reconstruct(A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, block_size):
    grid = lambda meta: (
        ((m + block_size - 1) // block_size, (n + block_size - 1) // block_size),
    )
    svd_reconstruct_kernel[grid](A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, block_size)

def fused_svd_reconstruct_torch(A: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    block_size = 64  # Example block size, can be adjusted

    # Allocate memory for U, S, and Vh
    U = torch.zeros((m, n), device=A.device)
    S = torch.zeros(m, device=A.device)
    Vh = torch.zeros((m, n), device=A.device)

    # Convert tensors to pointers
    A_ptr = A.data_ptr()
    U_ptr = U.data_ptr()
    S_ptr = S.data_ptr()
    Vh_ptr = Vh.data_ptr()

    # Call the Triton kernel
    fused_svd_reconstruct(A_ptr, U_ptr, S_ptr, Vh_ptr, m, n, block_size)

    # Reconstruct the matrix A
    A_reconstructed = torch.matmul(U, torch.diag_embed(S))
    A_reconstructed = torch.matmul(A_reconstructed, Vh.conj().t())

    return A_reconstructed
