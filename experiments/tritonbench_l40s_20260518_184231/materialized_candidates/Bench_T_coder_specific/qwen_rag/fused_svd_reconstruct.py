import triton
import triton.language as tl
import torch

# Define the kernel for reconstructing the matrix using SVD components
@triton.jit
def svd_reconstruct_kernel(U, S, Vh, A_out, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    m_mask = off_m[:, None] < M
    n_mask = off_n[None, :] < N
    
    u_block = tl.load(U + off_m[:, None] * K + off_n[None, :], mask=m_mask & n_mask)
    s_block = tl.load(S + off_m, mask=off_m < M)
    vh_block = tl.load(Vh + off_n[None, :] * M + off_m[:, None], mask=n_mask & m_mask)
    
    A_val = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        A_val += u_block[:, :, k] * s_block[k] * vh_block[k]
    
    tl.store(A_out + off_m[:, None] * N + off_n[None, :], A_val, mask=m_mask & n_mask)

# Wrapper function to invoke the kernel and perform the computation
def fused_svd_reconstruct(A):
    # Perform SVD decomposition (assuming a function `svd` exists)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get the shapes of the input matrix and SVD components
    M, N = A.shape
    K = min(M, N)
    
    # Create output tensor for the reconstructed matrix
    A_out = torch.empty_like(A)
    
    # Configure the block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Calculate the grid size
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Launch the Triton kernel
    svd_reconstruct_kernel[(grid_m, grid_n, 1)](U, S, Vh, A_out, M, N, K, BLOCK_M, BLOCK_N)
    
    return A_out

# Example usage
if __name__ == "__main__":
    A = torch.randn(4, 3)
    A_reconstructed = fused_svd_reconstruct(A)
    print("Original Matrix:\n", A)
    print("Reconstructed Matrix:\n", A_reconstructed)
