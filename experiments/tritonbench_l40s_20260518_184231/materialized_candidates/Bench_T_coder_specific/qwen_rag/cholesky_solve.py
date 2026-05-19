import triton
import triton.language as tl
import torch

# Triton kernel to compute the solution of a system of linear equations using Cholesky decomposition
@triton.jit
def cholesky_solve_kernel(
    B_ptr,  # Pointer to the right-hand side tensor B
    L_ptr,  # Pointer to the lower triangular Cholesky decomposition L
    X_ptr,  # Pointer to the output tensor X
    M,      # Number of rows in B and X
    K,      # Number of columns in B and rows in X
    N,      # Number of columns in L
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_K: tl.constexpr,  # Block size for columns
):
    pid = tl.program_id(0)
    bid = pid // (BLOCK_M * BLOCK_K)
    i0 = pid % BLOCK_M
    j0 = pid // BLOCK_M
    b = bid // (BLOCK_M * BLOCK_K)
    
    # Initialize shared memory buffers
    X_shared = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    Y_shared = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # Load B into shared memory
    for k in range(0, K, BLOCK_K):
        X_shared[i0, k:k+BLOCK_K] = tl.load(B_ptr + ((b * M + i0) * K + (k + j0)), mask=(i0 < M) & (k < K))
        tl.barrier(tl.memory_order.sync)
    
    # Solve LY = B
    for j in range(j0, N, BLOCK_K):
        for k in range(k0, j, BLOCK_K):
            y = X_shared[i0, k:k+BLOCK_K].sum(axis=1)
            for l in range(k+1, j+1):
                y -= L_ptr[((b * N + k) * N + l) * M + i0] * X_shared[i0, l:l+BLOCK_K].sum(axis=1)
            Y_shared[i0, j:j+BLOCK_K] = y
    
    # Store Y back to global memory
    for k in range(0, K, BLOCK_K):
        tl.store(X_ptr + ((b * M + i0) * K + (k + j0)), Y_shared[i0, k:k+BLOCK_K], mask=(i0 < M) & (k < K))
        tl.barrier(tl.memory_order.sync)
    
    # Solve L^T X = Y
    for j in range(N-1, j0, -BLOCK_K):
        for k in range(K-1, k0, -BLOCK_K):
            x = Y_shared[i0, k:k+BLOCK_K].sum(axis=1)
            for l in range(k+1, N):
                x -= L_ptr[((b * N + l) * N + j) * M + i0] * Y_shared[i0, l:l+BLOCK_K].sum(axis=1)
            X_shared[i0, k:k+BLOCK_K] = x
    
    # Store X back to global memory
    for k in range(0, K, BLOCK_K):
        tl.store(X_ptr + ((b * M + i0) * K + (k + j0)), X_shared[i0, k:k+BLOCK_K], mask=(i0 < M) & (k < K))
