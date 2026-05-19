import triton
import triton.language as tl
import torch

@triton.jit
def batched_vecmat_kernel(
    A, B, O,
    dim_m, dim_n, dim_k,
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    n_block_idx = tl.program_id(1)
    m_range = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    vecmat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of BLOCK_K segments to iterate over
    k_blocks = (dim_k + BLOCK_K - 1) // BLOCK_K
    for k_block_idx in range(k_blocks):
        k_range = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        # Load A for this block (shape [BLOCK_M, BLOCK_K])
        a_ptrs = A + (m_range[:, None] * stride_am) + (k_range[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(m_range[:, None] < dim_m) & (k_range[None, :] < dim_k), other=0.0)
        # Load B for this block (shape [BLOCK_M, BLOCK_N, BLOCK_K] => but we index each M separately)
        b_ptrs = B + (m_range[:, None, None] * stride_bm) + (n_range[None, :, None] * stride_bn) + (k_range[None, None, :] * stride_bk)
        b = tl.load(b_ptrs, mask=(m_range[:, None, None] < dim_m) & (n_range[None, :, None] < dim_n) & (k_range[None, None, :] < dim_k), other=0.0)
        # Broadcast A to match B in shape for multiplication
        a_broadcast = tl.broadcast_to(a[:, None, :], b.shape)
        # Sum over K dimension
        partial = tl.sum(a_broadcast * b, axis=2)
        vecmat += partial

    # Store results
    o_ptrs = O + (m_range[:, None] * stride_om) + (n_range[None, :] * stride_on)
    tl.store(o_ptrs, vecmat, mask=(m_range[:, None] < dim_m) & (n_range[None, :] < dim_n))

def batched_vecmat(A, B, block_m=32, block_n=32, block_k=32):
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device."
    dim_m, dim_k = A.shape
    assert B.shape[0] == dim_m, "M dimension of A must match first dimension of B."
    dim_n = B.shape[1]
    assert B.shape[2] == dim_k, "K dimension of A must match last dimension of B."
    assert dim_m % block_m == 0, "dim_m must be divisible by block_m."
    assert dim_n % block_n == 0, "dim_n must be divisible by block_n."
    assert dim_k % block_k == 0, "dim_k must be divisible by block_k."

    O = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)

    grid = (dim_m // block_m, dim_n // block_n)
    batched_vecmat_kernel[grid](
        A, B, O,
        dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        O.stride(0), O.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )
    return O
