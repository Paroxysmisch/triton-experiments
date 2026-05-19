import triton
import triton.language as tl
import torch

# Define the Triton kernel for matrix multiplication and symmetric update
@triton.jit
def matrix_multiply_symmetric_kernel(
    A_ptr, B_ptr, C_ptr,
    n, m, p, alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        A_block = tl.load(A_ptr + row[:, None] * m + k[None, :])
        B_block = tl.load(B_ptr + k[:, None] * p + col[None, :])
        acc += alpha * tl.dot(A_block, B_block)
    
    C_block = tl.load(C_ptr + row[:, None] * p + col[None, :], mask=(row[:, None] < n) & (col[None, :] < p))
    C_block = alpha * C_block @ C_block.T + beta * C_block
    
    tl.store(C_ptr + row[:, None] * p + col[None, :], C_block, mask=(row[:, None] < n) & (col[None, :] < p))

# Wrapper function to call the Triton kernel
def matrix_multiply_symmetric(A, B, C, alpha, beta):
    assert A.shape == (A.size(0), A.size(1))
    assert B.shape == (B.size(0), B.size(1))
    assert C.shape == (C.size(0), C.size(1))
    assert A.size(1) == B.size(0)
    assert C.size(0) == C.size(1)
    
    n, m = A.shape
    m, p = B.shape
    
    block_size_m = 32
    block_size_n = 32
    block_size_k = 8
    
    grid_m = (n + block_size_m - 1) // block_size_m
    grid_n = (p + block_size_n - 1) // block_size_n
    
    # Launch the Triton kernel
    matrix_multiply_symmetric_kernel[(grid_m, grid_n)](
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        n, m, p, alpha, beta,
        BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_K=block_size_k
    )
    
    return C
