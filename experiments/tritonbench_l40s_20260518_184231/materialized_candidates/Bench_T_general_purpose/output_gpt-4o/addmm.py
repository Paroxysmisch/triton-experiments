import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    mat1_ptr, mat2_ptr, input_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dm, stride_dn,
    alpha, beta,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate the start of the block
    row_start = pid // (N // BLOCK_N) * BLOCK_M
    col_start = pid % (N // BLOCK_N) * BLOCK_N
    
    # Initialize accumulation registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks of mat1 and mat2
        a = tl.load(mat1_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak)
        b = tl.load(mat2_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_bn)
        
        # Accumulate matrix multiplication result
        acc += tl.dot(a, b)
    
    # Scale by alpha
    acc *= alpha
    
    # Load input, scale by beta, and add to acc
    if beta != 0:
        c = tl.load(input_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_cm + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_cn)
        acc += beta * c
    
    # Store the result
    tl.store(out_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_dm + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_dn, acc)

import torch

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Determine the shapes
    M, K = mat1.shape
    _, N = mat2.shape
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    
    # Handle the sparse input case (not implemented in this example)
    # For now, we assume dense inputs
    
    # Launch the Triton kernel
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    addmm_kernel[grid](
        mat1, mat2, input, out,
        M, N, K,
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        input.stride(0), input.stride(1),
        out.stride(0), out.stride(1),
        alpha, beta,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out
