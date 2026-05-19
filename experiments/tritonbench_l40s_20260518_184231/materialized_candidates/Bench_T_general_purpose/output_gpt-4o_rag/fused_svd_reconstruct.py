import torch
import triton
import triton.language as tl

# Kernel function: Performs matrix multiplication C = A @ B
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_SIZE: tl.constexpr):
    # Calculate row and column indices for each thread
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create accumulators for each thread
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE):
        # Load blocks of A and B
        a_block = tl.load(A + row[:, None] * K + (k + tl.arange(0, BLOCK_SIZE)), mask=(row[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE) < K), other=0.0)
        b_block = tl.load(B + (k + tl.arange(0, BLOCK_SIZE))[:, None] * N + col, mask=(k + tl.arange(0, BLOCK_SIZE)[:, None] < K) & (col < N), other=0.0)

        # Compute matrix multiplication for the block
        acc += tl.dot(a_block, b_block)

    # Store the result in C
    tl.store(C + row[:, None] * N + col, acc, mask=(row[:, None] < M) & (col < N))

# Wrapper function to perform SVD and reconstruct the matrix
def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Perform SVD using PyTorch
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Convert S to a diagonal matrix
    S_diag = torch.diag(S)

    # Prepare output tensor for the result
    A_reconstructed = torch.empty_like(A)

    # Get matrix dimensions
    M, N = A.shape
    K = S_diag.shape[0]

    # Define block size
    BLOCK_SIZE = 32  # Example block size, adjust as needed

    # Launch Triton kernel for matrix multiplication U @ S_diag
    C_temp = torch.empty((M, K), dtype=torch.float32, device=A.device)
    grid_U_S = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(K, BLOCK_SIZE))
    matmul_kernel[grid_U_S](U, S_diag, C_temp, M, K, K, BLOCK_SIZE)

    # Launch Triton kernel for matrix multiplication (U @ S_diag) @ Vh
    grid_C_Vh = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    matmul_kernel[grid_C_Vh](C_temp, Vh, A_reconstructed, M, N, K, BLOCK_SIZE)

    return A_reconstructed
