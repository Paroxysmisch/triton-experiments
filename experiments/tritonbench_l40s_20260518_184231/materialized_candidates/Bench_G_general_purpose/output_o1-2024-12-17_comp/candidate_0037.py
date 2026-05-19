import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A_ptr, B_ptr, C_ptr,
    dim_m, dim_n, dim_k,
    strideAm, strideAk,
    strideBm, strideBn, strideBk,
    strideCm, strideCn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)

    m_range = tl.arange(0, BLOCK_M) + m_index * BLOCK_M
    n_range = tl.arange(0, BLOCK_N) + n_index * BLOCK_N

    vecmat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_blocks = dim_k // BLOCK_K
    for kb in range(k_blocks):
        k_range = tl.arange(0, BLOCK_K) + kb * BLOCK_K

        a_ptrs = A_ptr + (m_range[:, None] * strideAm) + (k_range[None, :] * strideAk)
        a = tl.load(a_ptrs, mask=(m_range[:, None] < dim_m) & (k_range[None, :] < dim_k), other=0.0)

        b_ptrs = B_ptr + (m_range[:, None, None] * strideBm) \
                          + (n_range[None, :, None] * strideBn) \
                          + (k_range[None, None, :] * strideBk)
        b = tl.load(b_ptrs,
                    mask=(m_range[:, None, None] < dim_m)
                         & (n_range[None, :, None] < dim_n)
                         & (k_range[None, None, :] < dim_k),
                    other=0.0)
        vecmat += tl.sum(a[:, None, :] * b, axis=2)

    c_ptrs = C_ptr + (m_range[:, None] * strideCm) + (n_range[None, :] * strideCn)
    tl.store(c_ptrs, vecmat, mask=(m_range[:, None] < dim_m) & (n_range[None, :] < dim_n))


def batched_vecmat(A, B, block_m=64, block_n=64, block_k=32):
    assert A.dim() == 2
    assert B.dim() == 3
    dim_m, dim_k = A.shape
    dim_m2, dim_n, dim_k2 = B.shape
    assert dim_m == dim_m2 and dim_k == dim_k2, "Incompatible shapes for A and B"
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m"
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n"
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k"

    A_gpu = A.to(device="cuda", dtype=torch.float32)
    B_gpu = B.to(device="cuda", dtype=torch.float32)
    C_gpu = torch.empty((dim_m, dim_n), device="cuda", dtype=torch.float32)

    grid = (dim_m // block_m, dim_n // block_n)
    batched_vecmat_kernel[grid](
        A_gpu, B_gpu, C_gpu,
        dim_m, dim_n, dim_k,
        A_gpu.stride(0), A_gpu.stride(1),
        B_gpu.stride(0), B_gpu.stride(1), B_gpu.stride(2),
        C_gpu.stride(0), C_gpu.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )
    return C_gpu.cpu()
