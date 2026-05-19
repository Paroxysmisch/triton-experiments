import triton
import triton.language as tl

@triton.jit
def svd_reconstruct_kernel(
    A_ptr,  # Pointer to the input matrix A
    U_ptr,  # Pointer to the U matrix from SVD
    S_ptr,  # Pointer to the singular values S
    Vh_ptr, # Pointer to the Vh matrix from SVD
    A_reconstructed_ptr,  # Pointer to the reconstructed matrix A
    M,  # Number of rows in A
    N,  # Number of columns in A
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the range of rows and columns for this block
    row_start = block_start
    row_end = tl.minimum(row_start + BLOCK_SIZE, M)
    col_start = block_start
    col_end = tl.minimum(col_start + BLOCK_SIZE, N)

    # Load the submatrices
    A_sub = tl.load(A_ptr + row_start * N + col_start, mask=row_start < M and col_start < N, other=0.0)
    U_sub = tl.load(U_ptr + row_start * N + col_start, mask=row_start < M and col_start < N, other=0.0)
    S_sub = tl.load(S_ptr + col_start, mask=col_start < N, other=0.0)
    Vh_sub = tl.load(Vh_ptr + col_start * N + col_start, mask=col_start < N, other=0.0)

    # Compute the reconstructed submatrix
    A_reconstructed_sub = tl.dot(U_sub, tl.diag(S_sub)) @ Vh_sub

    # Store the reconstructed submatrix
    tl.store(A_reconstructed_ptr + row_start * N + col_start, A_reconstructed_sub, mask=row_start < M and col_start < N)

import torch
import triton
import triton.language as tl

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the same device as the Triton kernel
    device = A.device
    assert device.type == 'cuda', "Input tensor must be on a CUDA device"

    # Compute the SVD of A
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # Create the reconstructed matrix
    A_reconstructed = torch.zeros_like(A, device=device)

    # Define the grid and block sizes
    M, N = A.shape
    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))

    # Launch the Triton kernel
    svd_reconstruct_kernel[grid](
        A, U, S, Vh, A_reconstructed,
        M, N, BLOCK_SIZE
    )

    return A_reconstructed

# Sample input matrix
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')

# Reconstruct the matrix
A_reconstructed = fused_svd_reconstruct(A)

# Print the original and reconstructed matrices
print("Original Matrix A:")
print(A)
print("Reconstructed Matrix A:")
print(A_reconstructed)
