import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block offsets
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    
    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the A and B blocks
        a_block = tl.load(a_ptr + rm * stride_am + k * stride_ak, mask=rm + tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
        b_block = tl.load(b_ptr + k * stride_bk + rn * stride_bn, mask=rn + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
        
        # Perform the matrix multiplication
        accumulator += tl.dot(a_block, b_block)
    
    # Store the result in the C matrix
    c_block = accumulator
    tl.store(c_ptr + rm * stride_cm + rn * stride_cn, c_block, mask=(rm + tl.arange(0, BLOCK_SIZE_M) < M) & (rn + tl.arange(0, BLOCK_SIZE_N) < N))

import torch

def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Get the dimensions of the matrices
    M, K = a.shape
    K, N = b.shape
    
    # Define the block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Define the grid and block sizes
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    block = (BLOCK_SIZE_M, BLOCK_SIZE_N, 1)
    
    # Define the strides
    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)
    
    # Launch the kernel
    matmul_kernel[grid, block](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
