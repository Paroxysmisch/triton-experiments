import torch
import triton
import triton.language as tl

# Kernel function: Computes the SVD and reconstructs the matrix
@triton.jit
def svd_reconstruct_kernel(A, U, S, Vh, m, n, BLOCK_SIZE: tl.constexpr):
    # Calculate the row and column indices for the output matrix
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.arange(0, n)
    
    # Determine which elements are valid within the block
    mask = row < m
    
    # Load U and Vh
    U_value = tl.load(U + row[:, None], mask=mask)
    Vh_value = tl.load(Vh + col[None, :], mask=mask)
    
    # Reconstruct the matrix A_reconstructed = U * diag(S) * Vh
    S_diag = tl.load(S + row, mask=mask)  # Load singular values
    A_reconstructed = tl.dot(U_value, S_diag[:, None] * Vh_value)
    
    # Store the result back to the output matrix
    tl.store(A + row[:, None] * n + col[None, :], A_reconstructed, mask=mask)

# Wrapper function to invoke the Triton kernel
def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    # Prepare output tensor with the same shape as A
    A_reconstructed = torch.empty_like(A)
    
    # Compute SVD using PyTorch
    U, S, Vh = torch.svd(A)
    
    # Launch the Triton kernel
    block_size = 32  # Define a suitable block size
    grid_size = (m + block_size - 1) // block_size
    svd_reconstruct_kernel[(grid_size, 1, 1)](A_reconstructed, U, S, Vh, m, n, block_size)
    
    return A_reconstructed
