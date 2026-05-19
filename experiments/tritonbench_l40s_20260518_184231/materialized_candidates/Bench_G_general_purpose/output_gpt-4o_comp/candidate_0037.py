import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(
    A_ptr, B_ptr, C_ptr,
    dim_m, dim_n, dim_k,
    block_m, block_n, block_k,
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_cm, stride_cn,
    **meta
):
    # Program ID
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    # Block indices
    m_start = m_index * block_m
    n_start = n_index * block_n

    # Create pointers for A, B, and C
    a_ptrs = A_ptr + m_start * stride_am + tl.arange(0, block_k) * stride_ak
    b_ptrs = B_ptr + m_start * stride_bm + n_start * stride_bn + tl.arange(0, block_k)[:, None] * stride_bk
    c_ptrs = C_ptr + m_start * stride_cm + n_start * stride_cn

    # Initialize accumulator
    vecmat = tl.zeros((block_m, block_n), dtype=tl.float32)

    # Loop over k dimension
    for k in range(0, dim_k, block_k):
        # Load slices of A and B
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Compute dot product for this block
        vecmat += tl.sum(a[:, None] * b, axis=0)

        # Update pointers for next block
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    # Store the result
    tl.store(c_ptrs, vecmat)


def batched_vecmat(A, B, block_m=64, block_n=64, block_k=64):
    assert A.shape[0] == B.shape[0], "Dimension mismatch: A and B must have the same first dimension."
    dim_m, dim_k = A.shape
    _, dim_n, _ = B.shape

    # Check dimensions
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k"
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m"
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n"

    # Output tensor
    C = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)

    # Grid dimensions
    grid = (dim_m // block_m, dim_n // block_n)

    # Launch kernel
    batched_vecmat_kernel[grid](
        A, B, C,
        dim_m, dim_n, dim_k,
        block_m, block_n, block_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1),
    )

    return C

# Example usage
dim_m, dim_k, dim_n = 128, 64, 128
A = torch.randn((dim_m, dim_k), device='cuda', dtype=torch.float32)
B = torch.randn((dim_m, dim_n, dim_k), device='cuda', dtype=torch.float32)

C = batched_vecmat(A, B)
print(C)
