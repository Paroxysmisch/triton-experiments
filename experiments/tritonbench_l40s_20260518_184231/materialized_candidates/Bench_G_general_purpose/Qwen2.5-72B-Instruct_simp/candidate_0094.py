import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block boundaries
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N
    
    # Initialize the output block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Compute the block boundaries for the K dimension
        rk = k
        a_block_ptr = a_ptr + (rm * stride_am + rk * stride_ak)
        b_block_ptr = b_ptr + (rk * stride_bk + rn * stride_bn)
        
        # Load the blocks from global memory to shared memory
        a_block = tl.load(a_block_ptr, mask=(rm + tl.arange(0, BLOCK_SIZE_M)) < M)
        b_block = tl.load(b_block_ptr, mask=(rn + tl.arange(0, BLOCK_SIZE_N)) < N)
        
        # Perform the dot product
        acc += tl.dot(a_block, b_block)
    
    # Store the result back to global memory
    c_block_ptr = c_ptr + (rm * stride_cm + rn * stride_cn)
    tl.store(c_block_ptr, acc, mask=(rm + tl.arange(0, BLOCK_SIZE_M)) < M)

import torch

def matmul(a, b, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16):
    # Get the dimensions of the input matrices
    M, K = a.shape
    K, N = b.shape
    
    # Allocate the output matrix
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Compute the grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return c
