import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(A, B, output, dim_m, dim_n, dim_k,
                          stride_am, stride_ak,
                          stride_bm, stride_bn, stride_bk,
                          stride_om, stride_on,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Program IDs for blocks
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_am[:, None] * stride_bm + offs_bn[None, :] * stride_bn + offs_k[None, :] * stride_bk

    # Accumulator for the block
    vecmat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension
    for k in range(0, tl.cdiv(dim_k, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < dim_k - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < dim_k - k * BLOCK_K, other=0.0)
        
        # Broadcasting a to match b's shape
        a_broadcasted = a[:, None, :]
        
        # Dot product for the block
        vecmat += tl.sum(a_broadcasted * b, axis=2)

        # Advance the pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write the result to the output
    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    o_ptrs = output + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    tl.store(o_ptrs, vecmat)

def batched_vecmat(A, B, block_m=64, block_n=64, block_k=64):
    dim_m, dim_k = A.shape
    _, dim_n, _ = B.shape

    # Ensure dimensions are divisible by block sizes
    assert dim_m % block_m == 0 and dim_n % block_n == 0 and dim_k % block_k == 0

    # Create output tensor
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)

    # Compute grid dimensions
    grid = (dim_m // block_m, dim_n // block_n)

    # Launch kernel
    batched_vecmat_kernel[grid](
        A, B, output,
        dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )

    return output
