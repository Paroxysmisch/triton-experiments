import torch
import triton
import triton.language as tl

# Triton kernel for batched vector-matrix multiplication
@triton.jit
def batched_vecmat_kernel(A, B, output, dim_m, dim_n, dim_k, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    m_index = tl.program_id(0)
    n_index = tl.program_id(1)
    k_index = tl.program_id(2)

    output_tile = tl.zeros([block_m, block_n], dtype=tl.float32)

    vecmat = 0

    k_blocks = tl.cdiv(dim_k, block_k)
    for k_block in range(k_blocks):
        a = tl.load(A + m_index * dim_k + k_block * block_k + tl.arange(0, block_k))
        b = tl.load(B + m_index * dim_n * dim_k + n_index * dim_k + k_block * block_k + tl.arange(0, block_k))
        broadcast = tl.broadcast(a[:, None], (block_n,))
        vecmat = tl.sum(broadcast * b[None, :], axis=0)
        output_tile += vecmat

    tl.store(output + m_index * dim_n + n_index + tl.arange(0, block_m) * dim_n + tl.arange(0, block_n)[:, None]
            , output_tile)

# Function to invoke the Triton kernel
def batched_vecmat(A, B):
    dim_m, dim_n, dim_k = A.shape[0], A.shape[1], B.shape[2]

    assert dim_m % 128 == 0
    assert dim_n % 64 == 0
    assert dim_k % 32 == 0

    output = torch.zeros((dim_m, dim_n), device=A.device, dtype=A.dtype)

    grid = lambda meta: (triton.cdiv(dim_m, meta["block_m"]), triton.cdiv(dim_n, meta["block_n"]), 1)

    batched_vecmat_kernel[grid](A, B, output, dim_m, dim_n, dim_k, block_k=32, block_m=128, block_n=64)

    return output
