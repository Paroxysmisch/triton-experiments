import triton
import triton.language as tl

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the block offsets
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_warp = num_pid_m * num_pid_n
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Initialize offsets for matrix A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to the blocks of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_pid_k):
        # Load the blocks of A and B
        if type == "default":
            a_block = tl.load(a_ptrs)
            b_block = tl.load(b_ptrs)
        elif type == "masked":
            a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
            b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K)
        elif type == "broadcast":
            a_block = tl.load(a_ptrs, broadcast=True)
            b_block = tl.load(b_ptrs, broadcast=True)
        else:
            raise ValueError("Invalid type")

        # Perform the matrix multiplication
        accumulator += tl.dot(a_block, b_block)

        # Update the pointers to the next block of A and B
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Compute the offsets for the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Store the result in the output matrix C
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)

import triton
import triton.language as tl
import numpy as np
import torch

def iv_dependent_matmul_wrapper(a, b, type="default", BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16):
    # Convert inputs to Torch tensors
    a = torch.tensor(a, device='cuda', dtype=torch.float32)
    b = torch.tensor(b, device='cuda', dtype=torch.float32)
    
    # Dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    # Grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Invoke the kernel
    iv_dependent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        type
    )
    
    return c.cpu().numpy()
